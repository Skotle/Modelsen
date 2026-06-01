# Cloud Build

이 저장소는 세 가지 클라우드 빌드 경로를 제공합니다.

## GitHub Actions

`.github/workflows/cloud-build.yml`은 push, pull request, manual dispatch에서 다음을 실행합니다.

1. 의존성 설치
2. `python -m unittest`
3. 합성 train/validation corpus 빌드
4. 8 step smoke 학습
5. validation loss 평가
6. checkpoint와 manifest artifact 업로드
7. Docker image build smoke

## Google Cloud Build

`cloudbuild.yaml`을 사용합니다.

```powershell
gcloud builds submit --config cloudbuild.yaml .
```

기본 artifact 경로는 `gs://$PROJECT_ID-irisen-artifacts/$BUILD_ID`입니다. 해당 버킷은 미리 만들어져 있어야 합니다.

## AWS CodeBuild

`buildspec.yml`을 사용합니다. CodeBuild project의 source root에 이 저장소를 연결하면 같은 smoke build가 실행됩니다.

## Local Equivalent

클라우드 runner가 실행하는 핵심 명령은 로컬에서도 같습니다.

```powershell
python -m unittest
python scripts/build_corpus.py --out-dir data/build --train-examples 512 --val-examples 128 --eval-examples 16
python -m irisen_model.train --config configs/irisen-local-build.json --data data/build/irisen_train.txt --val-data data/build/irisen_val.txt --out runs/cloud_smoke.pt --steps 8 --eval-interval 4 --eval-batches 2 --batch-size 4 --context-size 64 --d-model 48 --layers 2 --heads 4
python scripts/evaluate.py --checkpoint runs/cloud_smoke.pt --data data/build/irisen_val.txt --batches 4 --batch-size 4
```

