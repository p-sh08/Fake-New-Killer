import streamlit as st
import pickle

# 1. 완벽하게 동기화된 진짜 모델 불러오기
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf.pkl', 'rb') as f:
        tfidf = pickle.load(f)
except Exception as e:
    st.error(f"⚠️ 모델 로드 실패: {str(e)}")

st.title("🤖 AI 가짜 뉴스 검출 시스템 (Real AI v5)")
st.markdown("---")
st.write("500개 데이터셋과 완벽히 동기화된 TF-IDF 사전 기반의 실시간 AI 판별기입니다.")

# 뉴스 입력창
news_text = st.text_area("뉴스 기사 입력", placeholder="여기에 뉴스 기사를 붙여넣으세요...", height=200)

if st.button("AI 모델로 판별하기"):
    if news_text.strip() == "":
        st.warning("⚠️ 뉴스 기사를 입력해주세요!")
    else:
        try:
            # 💡 [순수 AI 연산] 코랩과 100% 일치하는 단어 변환 및 확률 예측
            text_vector = tfidf.transform([news_text])
            prob = model.predict_proba(text_vector)[0]
            
            # label 0 = 가짜뉴스 / label 1 = 진짜뉴스
            fake_prob = prob[0] * 100
            real_prob = prob[1] * 100
            
            # 결과 출력
            st.subheader("📊 AI 분석 결과")
            st.write(f"- **진짜 뉴스일 확률:** {real_prob:.1f}%")
            st.write(f"- **가짜 뉴스일 확률:** {fake_prob:.1f}%")
            st.markdown("---")
            
            if fake_prob > 50:
                st.error("🚨 주의: 가짜 뉴스일 가능성이 높습니다!")
            else:
                st.success("🍏 확인: 신뢰할 수 있는 진짜 뉴스입니다.")
                
        except Exception as e:
            st.error(f"❌ 판별 오류: {str(e)}")
