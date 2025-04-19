import openai
import os
 
from tqdm import tqdm
from typing import List, Tuple
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from openai import OpenAI
 
import pandas as pd
# ---- Prompt Templates ----
def build_prompt(text: str, labels: List[str], template_type: int) -> str:
    if template_type == 1:
        return f"Text: {text}\nQuestion: What is the best label for this text? Options: {labels}\nAnswer:"
    elif template_type == 2:
        return f"Classify the following text into one of the categories {labels}.\n\nText: {text}\nCategory:"
    elif template_type == 3:
        # Few-shot style
        return (
            f"Example 1: The acting was wooden and the plot was dull. → boring\n"
            f"Example 2: The film was visually stunning and creative. → artistic\n"
            f"Example 3: The story brought me to tears and felt so real. → emotional\n"
            f"Now classify:\n{text} →"
        )
    else:
        raise ValueError("Invalid template_type")

# ---- GPT Query Wrapper ----
def gpt_classify(prompt: str, temperature: float = 0.0) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    response =  client.chat.completions.create(model="gpt-4o-mini", messages=[{
        "role": "system",
        "content": "you are an helpful assistent."
    },
    {
        "role": "user",
        "content": prompt
    }])
    response_content = response.choices[0].message.content.strip().lower()
    print(response_content)
    text = response_content
    return text
 
# ==== Main Comparison Loop ====
def run_comparison(texts: List[str], labels: List[str], true_labels: List[str], output_csv: str, eval_txt: str):
    all_results = []
    eval_reports = []

    for template_type in [1, 2, 3]:
        print(f"\n🔍 Running with Template {template_type}...")
        preds = []
        confidences = []
        prompts = []
        responses = []

        for i, text in enumerate(tqdm(texts, desc=f"Template {template_type}")):
            prompt = build_prompt(text, labels, template_type)
            response = gpt_classify(prompt)

            best_label = max(labels, key=lambda l: response.count(l.lower()))
            confidence = 1.0 if best_label in response else 0.0

            preds.append(best_label)
            confidences.append(confidence)
            prompts.append(prompt)
            responses.append(response)

            all_results.append({
                "template_type": template_type,
                "input_text": text,
                "true_label": true_labels[i],
                "prompt": prompt,
                "response": response,
                "predicted_label": best_label,
                "confidence": confidence,
            })

        acc = accuracy_score(true_labels, preds)
        report = classification_report(true_labels, preds, target_names=labels, digits=3, output_dict=False)
        matrix = confusion_matrix(true_labels, preds, labels=labels)

        # Add to eval report
        eval_reports.append(f"\n===== Evaluation for Template {template_type} =====\n")
        eval_reports.append(f"Accuracy: {acc:.3f}\n")
        eval_reports.append(report)
        eval_reports.append(f"\nConfusion Matrix:\n{matrix}\n")

        print(f"✅ Accuracy (Template {template_type}): {acc:.3f}")

    # Save all results to CSV
    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\n📁 Detailed results saved to: {output_csv}")

    # Save evaluation to TXT
    with open(eval_txt, "w") as f:
        for line in eval_reports:
            f.write(line if isinstance(line, str) else str(line))
            f.write("\n")

    print(f"📊 Evaluation summary saved to: {eval_txt}")


# ==== Example Run ====
if __name__ == "__main__":
    # Example test data
    texts = [
        "The visuals and soundtrack were stunning.",
        "I didn’t care about the characters or the ending.",
        "The performance moved me to tears.",
        "An innovative take on classic storytelling.",
        "Nothing special; just boring dialogue and tired tropes.",
    ]
    labels = ["artistic", "boring", "emotional"]
    true_labels = ["artistic", "boring", "emotional", "artistic", "boring"]

    output_csv = "zero_shot_results.csv"
    eval_file = "evaluation_report.txt"

    run_comparison(texts, labels, true_labels, output_csv, eval_file)