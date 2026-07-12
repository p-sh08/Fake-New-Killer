import streamlit as gr  # 이름만 gr로 맞춰서 기존 구조 유지
import pickle

# 1. 모델과 TF-IDF 벡터라이저 불러오기
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# 스트림릿 웹 화면 꾸미기
gr.title("🤖 AI 가짜 뉴스 검출 시스템")
gr.markdown("---")
gr.write("뉴스 기사를 아래에 입력하면 AI 모델이 진짜인지 가짜인지 확률을 분석합니다.")

# 텍스트 입력창
news_text = gr.text_area("뉴스 기사 입력", placeholder="여기에 뉴스 기사를 붙여넣으세요...", height=200)

# 판별 버튼
if gr.button("AI 모델로 판별하기"):
    if news_text.strip() == "":
        gr.warning("⚠️ 뉴스 기사를 입력해주세요!")
    else:
        try:
            # 2. 텍스트 변환 및 예측
            text_vector = tfidf.transform([news_text])
            prob = model.predict_proba(text_vector)[0]
            
            fake_prob = prob[0] * 100  # 가짜뉴스 확률
            real_prob = prob[1] * 100  # 진짜뉴스 확률
            
            # 3. 결과 화면에 출력
            gr.subheader("📊 AI 분석 결과")
            gr.write(f"- **진짜 뉴스일 확률:** {real_prob:.1f}%")
            gr.write(f"- **가짜 뉴스일 확률:** {fake_prob:.1f}%")
            gr.markdown("---")
            
            if fake_prob > 50:
                gr.error("🚨 주의: 가짜 뉴스일 가능성이 높습니다!")
            else:
                gr.success("🍏 확인: 신뢰할 수 있는 진짜 뉴스일 가능성이 높습니다.")
                
        except Exception as e:
            gr.error(f"❌ 오류 발생: {str(e)}")
