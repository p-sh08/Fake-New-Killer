import gradio as gr
import pickle

# 코랩에서 보낸 모델과 tfidf 불러오기
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

def predict_news(news_text):
    try:
        text_vector = tfidf.transform([news_text])
        prob = model.predict_proba(text_vector)[0]

        fake_prob = prob[0] * 100  # 가짜뉴스 확률
        real_prob = prob[1] * 100  # 진짜뉴스 확률

        result = f"📊 AI 분석 결과\n"
        result += f"- 진짜 뉴스일 확률: {real_prob:.1f}%\n"
        result += f"- 가짜 뉴스일 확률: {fake_prob:.1f}%\n\n"

        if fake_prob > 50:
            result += "🚨 주의: 가짜 뉴스일 가능성이 높습니다!"
        else:
            result += "🍏 확인: 신뢰할 수 있는 진짜 뉴스일 가능성이 높습니다."
        return result
    except Exception as e:
        return f"❌ 오류 발생: {str(e)}"

demo = gr.Interface(
    fn=predict_news,
    inputs=gr.Textbox(lines=5, placeholder="여기에 뉴스 기사를 붙여넣으세요...", label="뉴스 기사 입력"),
    outputs=gr.Textbox(label="AI 모델 판별 결과"),
    title="🤖 AI 가짜 뉴스 검출 시스템"
)

demo.launch()
