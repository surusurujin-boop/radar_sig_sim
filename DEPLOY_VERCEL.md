# Vercel 배포 가이드

## 개요

| 환경 | 역할 |
|------|------|
| **Vercel** | 웹 UI Demo — 설계, 데이터 탐색(Live), API |
| **로컬** | PyTorch 학습 Job, DATA export, Ablation |

Vercel 서버리스는 **PyTorch·장시간 학습·영구 SQLite**를 지원하지 않으므로,
학습 기능은 로컬 전용으로 분리되어 있습니다.

## 배포 방법

### 1. Vercel CLI

```bash
npm i -g vercel
cd c:\LIG_AGENT\Radar
vercel login
vercel          # 최초 배포 (Preview)
vercel --prod   # Production 배포
```

### 2. GitHub 연동

1. [vercel.com](https://vercel.com) → Import Git Repository
2. `surusurujin-boop/radar_sig_sim` 선택
3. Framework Preset: **Other**
4. Root Directory: `.` (기본값)
5. `vercel.json` 이 install/build/routes 를 자동 적용

## 프로젝트 구조 (Vercel)

```
api/index.py              ← Serverless Flask 진입점
vercel.json               ← 라우팅·installCommand
requirements-vercel.txt   ← 경량 의존성 (torch 제외)
src/runtime.py            ← Vercel/로컬 분기
src/data/pulse_generator.py  ← torch 없이 Live 데이터 생성
```

## Vercel에서 가능한 기능

- `/` 대시보드 (DB는 `/tmp` — 인스턴스마다 초기화)
- `/design` 아키텍처·알고리즘 설명
- `/explorer` **Live** 모의 데이터 PDW/IQ/스펙트럼 탐색
- `/api/model-info`, `/api/training-phases`, `/api/scenarios`

## Vercel에서 불가능한 기능

- **학습 Job 시작** (`POST /api/jobs` → 503)
- npz 대용량 데이터셋 로드 (git에 `.npz` 미포함)
- 체크포인트 저장

## 로컬 전체 기능 실행

```bash
pip install -r requirements.txt
python app.py
# http://127.0.0.1:5000
```

## 환경 변수 (선택)

| 변수 | 설명 |
|------|------|
| `VERCEL` | Vercel이 자동 설정 (`1`) |

## 트러블슈팅

- **`Total bundle size exceeds 500 MB`**
  - 원인: 루트 `requirements.txt`에 **PyTorch(~2–4GB)** 가 있어 Vercel이 함께 설치함
  - 해결: `api/requirements.txt`(경량)만 사용, 루트 `requirements.txt`는 `.vercelignore`로 제외
  - 로컬 학습: `pip install -r requirements.txt` (Vercel과 별도)

- **404 on /explorer**: 서버 재배포 후 캐시 삭제
- **학습 버튼 비활성**: 정상 (Demo 모드)
- **데이터 탐색 default 데이터셋**: npz 미배포 → **Live** 모드 사용
