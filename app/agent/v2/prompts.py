"""GROW V2 노드에서 사용하는 간결한 구조화 프롬프트."""

MODE_PROMPT = """미숙아 보호자의 입력을 분류하세요.
coaching은 생활 속 변화를 함께 계획하거나 반복적으로 실천할 필요가 있는 고민입니다.
information은 사실·기준·이유를 묻는 단순 정보 질문입니다.
JSON만 반환하세요: {{"mode":"coaching|information"}}
입력: {question}"""

SAFETY_PROMPT = """미숙아/신생아 보호자의 입력에서 즉시 의료기관 안내가 필요한 위험 신호를 판별하세요.
호흡 곤란, 청색증, 의식 저하, 경련, 심한 처짐, 수유 불가 등 긴급 가능성이 있으면 emergency=true입니다.
JSON만 반환하세요: {{"emergency":true|false}}
입력: {message}"""

GOAL_PROMPT = """보호자가 원하는 변화를 1~3일 안에 관찰 가능한 행동 목표로 정리하세요.
의료적 치료나 완치를 목표로 표현하지 마세요. JSON만 반환하세요.
{{"goal":"...","success_criteria":"...","time_horizon_days":1}}
최초 고민: {question}
보호자 답변: {answer}"""

GOAL_REVISION_PROMPT = """보호자의 수정 의견을 반영해 현재 목표를 다시 정리하세요.
수정을 요청하지 않은 내용은 가능한 한 유지하고, 의료적 치료나 완치를 목표로 표현하지 마세요.
JSON만 반환하세요.
{{"goal":"...","success_criteria":"...","time_horizon_days":1}}
최초 고민: {question}
현재 목표: {current_goal}
현재 성공 기준: {current_criteria}
현재 실행 기간: {current_days}일
보호자 수정 의견: {feedback}"""

REALITY_PROMPT = """보호자의 현재 문제를 사실 중심으로 한두 문장으로 요약하고 실행 제약을 추출하세요.
목표는 앞으로 도달하려는 희망 상태이며 현재 사실이 아닙니다.
현재 상황 요약에 목표가 이미 달성된 것처럼 표현하지 마세요.
‘그랬다’, ‘계속 그렇다’ 같은 표현은 최초 질문에 나타난 문제를 기준으로 해석하세요.
잘함/못함, 있음/없음 등의 부정과 방향을 반대로 바꾸지 마세요.
병원 방문이나 처방은 시도한 방법으로만 기록하고 효과를 임의로 판단하지 마세요.
사용자가 말하지 않은 기간, 횟수, 결과를 추가하지 마세요.
JSON만 반환하세요: {{"summary":"...","constraints":["..."]}}
최초 질문: {question}
희망 목표: {goal}
답변: {answer}"""

OPTIONS_PROMPT = """보호자의 질문, 목표, 현재 상황과 근거 문서를 종합해 다음 내용을 만드세요.
1. 보호자가 상황을 이해할 수 있는 의학적 배경 설명을 2~3문장으로 작성하세요.
2. 지금 시도할 수 있는 작고 안전한 행동 선택지 2~3개를 만드세요.
3. 각 선택지에는 질문, 목표, 현재 상황과 근거를 토대로 그 선택지를 만든 이유를 작성하세요.
의료적 진단이나 처방을 하지 말고, 근거 문서에 없는 내용을 의학적 사실처럼 단정하지 마세요.
근거가 부족하면 일반적인 관찰·기록 행동을 우선하세요.
JSON만 반환하세요:
{{"medical_context":"...","options":[{{"id":"option-1","label":"...","reason":"..."}}]}}
최초 질문: {question}
목표: {goal}
현재 상황: {reality}
근거 문서: {context}"""

OPTIONS_VALIDATION_PROMPT = """다음 행동 선택지들을 엄격하게 검수하세요.
모든 선택지가 아래 기준을 통과할 때만 valid=true로 판정하세요.
1. 현재 목표에 직접 도움이 되는 행동인가
2. 목표와 반대되는 행동이 아닌가
3. 사용자가 말하지 않은 증상이나 문제를 가정하지 않는가
4. 현재 상황 및 근거 문서로 제안 이유를 설명할 수 있는가
5. 의료적 진단이나 처방을 새로 만들지 않는가
6. 보호자가 일상생활에서 무리 없이 실행하고 결과를 관찰할 수 있는가
관련 주제라는 이유만으로 목표에 직접 도움이 되지 않는 행동을 통과시키지 마세요.
JSON만 반환하세요:
{{"valid":true,"issues":[{{"option_id":"option-1","codes":["..."],"feedback":"..."}}]}}
최초 질문: {question}
현재 목표: {goal}
현재 상황: {reality}
근거 문서: {context}
후보 결과: {candidate}"""

OPTIONS_REGENERATION_PROMPT = """이전 행동 선택지가 검수를 통과하지 못했습니다.
검수 피드백을 모두 반영해 전체 선택지를 다시 만드세요.
실패한 행동을 표현만 바꿔 반복하지 말고, 사용자가 말하지 않은 문제를 새로 가정하지 마세요.
모든 선택지는 현재 목표에 직접 도움이 되고, 현재 상황과 근거 문서로 이유를 설명할 수 있어야 합니다.
의료적 진단이나 처방을 하지 말고, 보호자가 일상생활에서 현실적으로 실행할 수 있는 행동만 제안하세요.
JSON만 반환하세요:
{{"medical_context":"...","options":[{{"id":"option-1","label":"...","reason":"..."}}]}}
최초 질문: {question}
현재 목표: {goal}
현재 상황: {reality}
근거 문서: {context}
이전 결과: {previous}
검수 결과: {validation}"""

PLAN_PROMPT = """선택한 행동을 일상생활에서 실행하고 결과를 관찰할 수 있는 계획으로 정리하세요.
JSON만 반환하세요: {{"when":"...","duration":"...","observation":"..."}}
목표: {goal}
현재 상황: {reality}
선택 행동: {action}"""

REVIEW_PROMPT = """GROW 코칭의 실천 결과를 평가해 정확히 한 경로를 고르세요.
COMPLETED=성공 기준 충족, ADJUST_WILL=효과는 있으나 계획 조정 필요,
CHANGE_OPTION=행동 자체가 실행 곤란/효과 없음, UPDATE_REALITY=새 상황 발견,
CHANGE_GOAL=목표 변경 희망, EMERGENCY=위험 신호입니다.
JSON만 반환하세요: {{"route":"...","barrier":"...","reason":"..."}}
목표: {goal}
성공 기준: {criteria}
행동: {action}
결과: {result}"""
