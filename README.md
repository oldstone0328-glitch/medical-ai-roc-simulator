# 의료 AI 판정 기준 실험

고등학교 3분 발표에서 판정 기준에 따라 정확도, 민감도, 특이도, 위양성(FP), 위음성(FN)이 어떻게 달라지는지 보여주는 한 페이지짜리 Streamlit 앱입니다.

환자 200명(암 20명)의 데이터는 모두 가상이며, 실제 진단에 사용할 수 없습니다. 같은 결과로 발표할 수 있도록 가상 데이터의 난수 seed를 고정했습니다.

## 설치

Python 3.11 또는 3.12가 필요합니다. 프로젝트 폴더에서 다음 명령을 실행합니다.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## macOS에서 실행

터미널에 다음 한 줄만 붙여 넣습니다.

```bash
/bin/bash "/Users/phonei/Documents/Codex/2026-07-12/ai-99-ai-ai-roc-curve/run_mac.command"
```

또는 Finder에서 `run_mac.command`를 더블클릭합니다. macOS가 차단하면 파일을 우클릭한 뒤 `열기`를 선택합니다.

## 직접 실행

```bash
.venv/bin/python -m streamlit run app.py
```

## 3분 발표 조작 순서

1. 처음 50% 기준에서 정확도와 FN·FP를 확인합니다.
2. 기준을 20%로 낮춰 FN이 감소하고 FP가 증가하는 모습을 보여줍니다.
3. 기준을 80%로 높여 FN이 증가하고 FP가 감소하는 모습을 보여줍니다.
4. ROC Curve 위의 현재 점이 함께 이동하는 것을 설명합니다.

앱을 종료하려면 실행한 터미널에서 `Control+C`를 누릅니다.
