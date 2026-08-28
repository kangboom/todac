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

REALITY_PROMPT = """보호자의 현재 상황을 사실 중심으로 한두 문장으로 요약하고 실행 제약을 추출하세요.
JSON만 반환하세요: {{"summary":"...","constraints":["..."]}}
목표: {goal}
답변: {answer}"""

OPTIONS_PROMPT = """미숙아 보호자가 선택할 수 있는 작고 안전한 행동 대안 2~3개를 만드세요.
의료적 진단이나 처방을 하지 말고, 아래 근거가 없으면 일반적인 관찰·기록 행동을 우선하세요.
JSON만 반환하세요: {{"options":[{{"id":"option-1","label":"..."}}]}}
목표: {goal}
현재 상황: {reality}
근거 문서: {context}"""

PLAN_PROMPT = """선택한 행동을 1~3일간 실행 가능한 계획으로 정리하세요.
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
