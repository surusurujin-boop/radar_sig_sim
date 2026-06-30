"""NumPy 기반 합성 펄스 생성 (torch 불필요 — Vercel·탐색기용)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .modulation_features import enrich_pulse_iq, mod_type_to_id
from .scenarios import PriModulation

NOISE_EMITTER_ID = -1


@dataclass
class EmitterProfile:
    emitter_id: int
    cf: float
    pw: float
    pa: float
    doa: float
    pri_mean: float
    mod_freq: float
    mod_type: str
    pri_modulation: PriModulation = PriModulation.STABLE
    stagger_ratio: float = 1.5


class SyntheticRadarPulseGenerator:
    """다중 방사원 펄스열 합성 (설계서 4.1~4.3)."""

    def __init__(
        self,
        iq_length: int = 256,
        spec_height: int = 64,
        spec_width: int = 64,
        snr_db: float = 15.0,
        pri_modulation: PriModulation = PriModulation.STABLE,
        seed: int | None = None,
    ) -> None:
        self.iq_length = iq_length
        self.spec_height = spec_height
        self.spec_width = spec_width
        self.snr_db = snr_db
        self.pri_modulation = pri_modulation
        self.rng = np.random.default_rng(seed)

    def _make_emitter_profiles(self, num_emitters: int) -> list[EmitterProfile]:
        mod_types = ["cw", "lfm", "nlfm", "fsk", "psk"]
        profiles = []
        for i in range(num_emitters):
            profiles.append(
                EmitterProfile(
                    emitter_id=i,
                    cf=self.rng.uniform(8e9, 12e9),
                    pw=self.rng.uniform(1e-6, 50e-6),
                    pa=self.rng.uniform(0.3, 1.0),
                    doa=self.rng.uniform(-60, 60),
                    pri_mean=self.rng.uniform(100e-6, 2000e-6),
                    mod_freq=self.rng.uniform(0.5e6, 5e6),
                    mod_type=mod_types[i % len(mod_types)],
                    pri_modulation=self.pri_modulation,
                )
            )
        return profiles

    def _next_pri(self, profile: EmitterProfile, pulse_idx: int, _phase: float) -> float:
        base = profile.pri_mean
        mod = profile.pri_modulation

        if mod == PriModulation.STABLE:
            return base * self.rng.uniform(0.9, 1.1)
        if mod == PriModulation.JITTER:
            return base * self.rng.uniform(0.5, 1.5)
        if mod == PriModulation.STAGGER:
            factor = profile.stagger_ratio if pulse_idx % 2 == 1 else 1.0
            return base * factor * self.rng.uniform(0.95, 1.05)
        if mod == PriModulation.SLIDING:
            sweep = 0.3 * np.sin(2 * np.pi * pulse_idx / 16)
            return base * (1.0 + sweep) * self.rng.uniform(0.95, 1.05)
        if mod == PriModulation.GROUP:
            group = (pulse_idx // 4) % 3
            return base * [1.0, 1.3, 0.7][group] * self.rng.uniform(0.95, 1.05)
        return base

    def _add_noise(self, i: np.ndarray, q: np.ndarray, pa: float) -> tuple[np.ndarray, np.ndarray]:
        signal_power = np.mean(i**2 + q**2)
        noise_power = signal_power / (10 ** (self.snr_db / 10))
        noise_std = np.sqrt(noise_power / 2)
        i = i + self.rng.normal(0, noise_std, size=i.shape)
        q = q + self.rng.normal(0, noise_std, size=q.shape)
        return i.astype(np.float32), q.astype(np.float32)

    def _generate_iq(self, profile: EmitterProfile) -> np.ndarray:
        t = np.linspace(0, profile.pw, self.iq_length, endpoint=False)
        phase = 2 * np.pi * profile.mod_freq * t
        mt = profile.mod_type

        if mt == "lfm":
            chirp_rate = profile.mod_freq / max(profile.pw, 1e-9)
            phase = 2 * np.pi * (0.5 * chirp_rate * t**2)
        elif mt == "nlfm":
            chirp_rate = profile.mod_freq / max(profile.pw, 1e-9)
            phase = 2 * np.pi * (
                0.5 * chirp_rate * t**2
                + 0.2 * chirp_rate * np.sin(4 * np.pi * t / max(profile.pw, 1e-9)) * t**2
            )
        elif mt == "fsk":
            n_hops = 4
            hop_len = max(1, self.iq_length // n_hops)
            freqs = self.rng.choice([0.5e6, 1.0e6, 1.5e6, 2.0e6], size=n_hops)
            freq_sig = np.repeat(freqs, hop_len + 1)[: self.iq_length]
            phase = 2 * np.pi * np.cumsum(freq_sig) * (profile.pw / self.iq_length)
        elif mt == "psk":
            n_chips = max(4, self.iq_length // 16)
            symbols = self.rng.integers(0, 4, size=n_chips)
            expanded = np.repeat(symbols, self.iq_length // n_chips + 1)[: self.iq_length]
            phase = phase + (np.pi / 2) * expanded

        i = profile.pa * np.cos(phase)
        q = profile.pa * np.sin(phase)
        i, q = self._add_noise(i, q, profile.pa)
        return np.stack([i, q], axis=0)

    def _generate_spectrum(self, iq: np.ndarray, profile: EmitterProfile) -> np.ndarray:
        spec = np.abs(np.fft.fftshift(np.fft.fft2(iq.reshape(1, -1))))
        spec = spec / (spec.max() + 1e-8)
        h, w = self.spec_height, self.spec_width
        out = np.zeros((h, w), dtype=np.float32)
        sh, sw = spec.shape
        out[: min(h, sh), : min(w, sw)] = spec[: min(h, sh), : min(w, sw)]
        out += 0.1 * ((profile.cf - 10e9) / 2e9)
        return out[np.newaxis, ...]

    def _generate_noise_pulse(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        cf = self.rng.uniform(8e9, 12e9)
        pw = self.rng.uniform(1e-6, 50e-6)
        pa = self.rng.uniform(0.1, 0.5)
        doa = self.rng.uniform(-90, 90)

        t = np.linspace(0, pw, self.iq_length, endpoint=False)
        phase = 2 * np.pi * self.rng.uniform(0.5e6, 5e6) * t
        i = pa * np.cos(phase)
        q = pa * np.sin(phase)
        i, q = self._add_noise(i, q, pa)
        iq = np.stack([i, q], axis=0)

        spec = np.abs(np.fft.fftshift(np.fft.fft2(iq.reshape(1, -1))))
        spec = spec / (spec.max() + 1e-8)
        h, w = self.spec_height, self.spec_width
        out = np.zeros((1, h, w), dtype=np.float32)
        sh, sw = spec.shape
        out[0, : min(h, sh), : min(w, sw)] = spec[: min(h, sh), : min(w, sw)]

        pdw = np.array([cf, pw, pa, doa, 0.0], dtype=np.float32)
        return pdw, iq, out, -1

    def _build_pulse(
        self, profile: EmitterProfile | None, toa: float, is_noise: bool
    ) -> dict[str, np.ndarray]:
        if is_noise:
            pdw, iq, spec, mod_id = self._generate_noise_pulse()
            pdw[4] = toa
        else:
            assert profile is not None
            iq = self._generate_iq(profile)
            spec = self._generate_spectrum(iq, profile)
            pdw = np.array([profile.cf, profile.pw, profile.pa, profile.doa, toa], dtype=np.float32)
            mod_id = mod_type_to_id(profile.mod_type)

        extras = enrich_pulse_iq(iq, self.spec_height, self.spec_width)
        return {
            "pdw": pdw,
            "iq": iq,
            "spectrum": spec,
            "iq_inst": extras["iq_inst"],
            "iq_tf": extras["iq_tf"],
            "mod_label": np.int64(mod_id),
        }

    def generate_interleaved_sequence(
        self,
        num_emitters: int = 3,
        pulses_per_emitter: int = 20,
        drop_rate: float = 0.1,
        noise_pulse_rate: float = 0.0,
    ) -> dict[str, np.ndarray]:
        profiles = self._make_emitter_profiles(num_emitters)
        events: list[tuple[float, int, int]] = []

        for profile in profiles:
            toa = 0.0
            for pulse_idx in range(pulses_per_emitter):
                if self.rng.random() >= drop_rate:
                    events.append((toa, profile.emitter_id, pulse_idx))
                toa += self._next_pri(profile, pulse_idx, toa)

        if noise_pulse_rate > 0 and events:
            max_toa = max(e[0] for e in events)
            num_noise = max(1, int(len(events) * noise_pulse_rate))
            for _ in range(num_noise):
                events.append((self.rng.uniform(0, max_toa), NOISE_EMITTER_ID, 0))

        events.sort(key=lambda x: x[0])
        profile_map = {p.emitter_id: p for p in profiles}

        pulses: list[dict[str, np.ndarray]] = []
        labels: list[int] = []
        for toa, emitter_id, _ in events:
            is_noise = emitter_id == NOISE_EMITTER_ID
            profile = None if is_noise else profile_map[emitter_id]
            pulse = self._build_pulse(profile, toa, is_noise)
            pulses.append(pulse)
            labels.append(NOISE_EMITTER_ID if is_noise else emitter_id)

        return {
            "pdw": np.stack([p["pdw"] for p in pulses]),
            "iq": np.stack([p["iq"] for p in pulses]),
            "spectrum": np.stack([p["spectrum"] for p in pulses]),
            "iq_inst": np.stack([p["iq_inst"] for p in pulses]),
            "iq_tf": np.stack([p["iq_tf"] for p in pulses]),
            "mod_labels": np.array([p["mod_label"] for p in pulses], dtype=np.int64),
            "labels": np.array(labels, dtype=np.int64),
        }


def normalize_pdw(pdw: np.ndarray) -> np.ndarray:
    out = pdw.copy()
    out[:, 0] = (out[:, 0] - 10e9) / 2e9
    out[:, 1] = np.log10(out[:, 1] * 1e6 + 1e-8)
    out[:, 3] = out[:, 3] / 90.0
    if len(out) > 1:
        toa_min = out[:, 4].min()
        out[:, 4] = (out[:, 4] - toa_min) / (out[:, 4].max() - toa_min + 1e-8)
    else:
        out[:, 4] = 0.0
    return out.astype(np.float32)
