from flask import Flask, render_template, request
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Исправляем название задачи на "sentiment-analysis" и исправляем опечатку в имени переменной
sentiment_analyzer = pipeline("sentiment-analysis",model="blanchefort/rubert-base-cased-sentiment")


tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/rugpt3medium_based_on_gpt2")
model = AutoModelForCausalLM.from_pretrained("sberbank-ai/rugpt3medium_based_on_gpt2")



app = Flask(__name__)


def generate_recommendation(mood):
    promt = f"hjhjjroovbzxl;rut][dow[vcbjshft]] {mood}. dfkhioutvcxxsdlw[p[gycb, vmcbfut hdrferdfsjy hdgjkl gdzkjlfgh"
    inputs = tokenizer(promt, return_tensors="pt")
    output = model.generate(
        **inputs,
        max_lenght=70,
        do_sample=True,
        top_p=0.9,
        temperature=0.9,
    )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text[len(promt):].strip()
@app.route('/', methods=['GET','POST'])
def index():
    recommendation = ""
    user_text = ""
    if request.method == "POST":
        user_text = request.form["message"]
        result = sentiment_analyzer(user_text)[0] 
        label = result["label"]
        
        if label == "POSITIVE":
            recommendation = " 🤡 🤡 🤡Ваш текст позитивный! Продолжайте в том же духе."
        elif label == "NEGATIVE":
            recommendation = " 🤡 🤡 🤡Ваш текст негативный. Возможно, стоит пересмотреть взгляд на ситуацию."
        else:
            recommendation = " 🤡 🤡 🤡Нейтральный текст. Все в порядке."
        ai_text = generate_recommendation (recommendation)
        ai_text = f"Настроение {recommendation}, \рекомендации:{ai_text}"
          
        
    return render_template('rec.html', recommendation=recommendation, user_text=user_text)

if __name__ == '__main__':
    app.run(debug=True)