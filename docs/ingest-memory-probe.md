# ingest 메모리 반환 실험

`INGEST_MEMORY_PROBE=true`일 때, 성공한 작업을 실행한 worker 프로세스에서만 진단합니다. 기본값은 꺼짐입니다. `INGEST_MEASUREMENTS_DIR`도 설정되어야 합니다.

## 측정 흐름

1. 파서 생성 직후와 파싱 완료 직후에 `weakref`로 파서·converter·pipeline·레이아웃 모델·표 모델을 등록합니다. 파싱 전에는 모델이 아직 없을 수 있습니다.
2. 기존 cleanup과 작업 함수 반환을 기다립니다. 기존 `task_end`는 반환 직후의 기준값으로 그대로 기록합니다.
3. `memory_probe / after_return`: 객체 생존 여부, RSS, 익명 RSS, PSS, Private_Dirty를 기록합니다.
4. 같은 프로세스에서 `gc.collect()` 후 `after_gc`를 기록합니다.
5. 같은 프로세스의 glibc `malloc_trim(0)` 후 `after_trim`을 기록합니다. Linux/glibc가 아니면 미지원으로 기록합니다.

세 단계 사이에는 await가 없습니다. asyncio 실행 흐름에서 다음 작업에 제어를 넘기지 않습니다. 별도 스레드에서 실행 중인 다른 작업까지 중지하는 기능은 아니므로 `--max-async-tasks 1`인 조건에서 비교합니다.
관측 객체는 문자열과 약한 참조로만 보관하며, 모델이나 문서 본문을 로그에 저장하지 않습니다. 실제 설치된 Docling 2.124.0의 객체 경로를 사용합니다. 다른 버전에서는 등록된 항목과 type을 먼저 확인하세요.

## 실행

서버의 backend 디렉터리에 있는 `.env`에 다음 설정을 추가합니다. 기존 worker의 `env_file`이 이 값을 읽습니다.

```env
INGEST_MEMORY_PROBE=true
```

새 코드가 포함된 이미지를 배포한 뒤 기존 Compose 파일로 worker를 재생성합니다. 이미지 빌드·업로드는 별도 배포 절차를 따릅니다. `.env` 수정만으로 이미 실행 중인 worker에 설정이나 새 코드가 적용되지는 않습니다.

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --no-deps --force-recreate worker
```

worker 재생성이 필요하므로 진행 중인 ingest가 끝난 뒤 적용합니다. 이전 worker에 남아 있던 메모리를 소급 측정하는 방식이 아니라, 새 코드로 다음 ingest가 끝났을 때 측정합니다. worker 수와 3 GiB 한도는 선택한 기존 Compose 파일을 따릅니다. 이미 별도로 바꾼 worker 수의 효과와 이번 GC/trim 효과는 구분해서 기록합니다.

PDF 하나를 업로드해 완료를 기다린 뒤:

```bash
docker logs --since 10m todac-worker 2>&1 | grep '"event": "memory_probe"'
```

기존 `/var/log/todac/ingest/*.jsonl`에도 동일한 이벤트가 저장됩니다. `doc_id`, `attempt_id`, `pid`가 같은 이벤트의 세 checkpoint를 비교합니다. 추가 진단은 `task_end` 뒤에 기록되므로 `after_trim` 또는 `memory_probe_error`까지 확인합니다. `objects_tracked`는 등록 시점 기록입니다.

실험 종료 후 `.env`에서 `INGEST_MEMORY_PROBE=false`로 바꾸거나 해당 항목을 제거하고 worker를 재생성하면 꺼집니다.

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --no-deps --force-recreate worker
```

## 읽는 방법

- 메모리 값은 bytes입니다. `1048576`으로 나누면 MiB입니다.
- `objects`의 `alive=true`는 해당 객체가 아직 살아 있다는 뜻입니다. `false`는 수거됐다는 뜻입니다. `null`은 약한 참조를 지원하지 않아 판정할 수 없다는 뜻입니다. 등록되지 않은 항목을 죽은 객체로 해석하지 않습니다.
- `after_return`에서 살아 있던 객체가 `after_gc`에서 사라지면, 함수 반환 뒤에 남은 순환 참조 등 GC 대상이었을 가능성이 있습니다. 정확한 참조 경로는 별도 조사 대상입니다.
- `rss_released_since_previous_bytes`가 양수면 직전 단계보다 RSS가 감소했습니다. 음수면 증가한 것입니다.
- 객체가 사라진 뒤 `after_trim`에서 RSS가 크게 줄면 glibc가 보관하던 반환 가능한 메모리가 기여했습니다. trim 성공 코드 1은 반환한 바이트 수가 아닙니다.
- 모델 소유 객체의 소멸이 모든 tensor storage나 네이티브 버퍼의 소멸을 증명하지는 않습니다. 줄지 않는 잔여량 전체를 누수라고 부르지 않습니다.
- GC 수거 개수는 객체 개수이며 메모리 바이트 수가 아닙니다. 프로세스 누적 최대 RSS는 내려가지 않으므로 효과 판정에는 현재 RSS/PSS를 사용합니다.
- 진단은 메모리 상태를 바꾸는 실험입니다. `task_end`는 개입 전, `after_gc`/`after_trim`은 개입 후 값입니다. 다음 회차의 시작 조건도 달라집니다.
- 예외가 밖으로 전파되거나 저장 완료 표시가 없는 작업에는 반환 후 실험을 수행하지 않습니다. OOM/SIGKILL은 Python 정리 코드를 실행하지 못하므로 세 checkpoint도 없습니다.

## 로컬 검증

무거운 모델·DB 없이 객체 생존과 호출 순서를 검증합니다. 실제 RSS 반환량은 Linux 서버 ingest에서 확인해야 합니다.

```bash
python -m unittest discover -s tests -p test_ingest_memory_probe.py
```
