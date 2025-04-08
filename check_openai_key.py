import os
from openai import OpenAI, AuthenticationError

def check_openai_key():
    try:
            # 환경 변수에서 키를 가져옵니다.
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("오류: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        else:
            # print(f"DEBUG: 사용 중인 API 키 시작 부분: {api_key[:5]}...") # 디버깅용 (실제 키 노출 주의)
            client = OpenAI(api_key=api_key)

            # 간단한 모델 목록 조회 테스트
            print("모델 목록 조회를 시도합니다...")
            models = client.models.list()
            print("모델 목록 조회 성공!")
            # print(list(models)) # 실제 모델 목록 출력 (선택 사항)

            # 원래 번역 코드 테스트 (모델 목록 조회가 성공한 경우)
            print("\n번역 테스트를 시도합니다...")
            text_to_translate = "안녕하세요"
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional translator. Translate the given Korean text to English accurately and naturally."},
                    {"role": "user", "content": text_to_translate}
                ]
            )
            print("번역 결과:", response.choices[0].message.content.strip())

    except AuthenticationError as e:
        print(f"OpenAI 인증 오류 발생: {e}")
        print("API 키 값, 환경 변수 설정, OpenAI 계정 상태(결제, 한도)를 확인하세요.")
    except Exception as e:
        print(f"예상치 못한 오류 발생: {e}")


if __name__ == "__main__":
    check_openai_key()