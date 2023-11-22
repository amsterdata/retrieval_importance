from datasets import load_dataset
from manifest import Manifest
import numpy as np

web_questions_train = load_dataset("web_questions", split="train")
web_questions_test = load_dataset("web_questions", split="test")

questions = web_questions_train['question'] + web_questions_test['question']
answers = web_questions_train['answers'] + web_questions_test['answers']

qa_pairs = list(zip(questions, answers))

gpt35 = Manifest(client_name="openai", engine="text-davinci-003",
                 cache_name="sqlite", cache_connection="gpt35-webq.sqlite")

from sklearn.model_selection import train_test_split

accuracies = []

for i in range(0, 64):
    _, test = train_test_split(qa_pairs, random_state=i, test_size=0.5)

    num_questions = 0
    num_correct = 0

    for question, correct_answers in test:

        few_shot = "Question: what is the name of justin bieber brother?\nAnswer: Jazmyn Bieber\n\nQuestion: what country is the grand bahama island in?\nAnswer: Bahamas\n\nQuestion: who did draco malloy end up marrying?\nAnswer:  Astoria Greengrass\n\nQuestion: what character did natalie portman play in star wars?\nAnswer: Padmé Amidala\n\nQuestion: what kind of money to take to bahamas?\nAnswer: Bahamian dollar\n\nQuestion: what character did john noble play in lord of the rings?\nAnswer:  Denethor II\n\nQuestion: who does joakim noah play for?\nAnswer: Chicago Bulls\n\nQuestion: where are the nfl redskins from?\nAnswer: Washington Redskins\n\nQuestion: what two countries invaded poland in the beginning of ww2?\nAnswer:  Germany\n\nQuestion: what is nina dobrev nationality?\nAnswer: Bulgaria\n\nQuestion: who did the philippines gain independence from?\nAnswer: United States of America\n\nQuestion: what are the major cities in france?\nAnswer:  Paris\n\nQuestion: where to fly into bali?\nAnswer: Ngurah Rai Airport\n\nQuestion: who is the prime minister of ethiopia?\nAnswer: Hailemariam Desalegn\n\nQuestion: what high school did president bill clinton attend?\nAnswer:  Hot Springs High School\n\n"

        prompt = f'{few_shot}Question: {question}?\nAnswer: '

        generated_answer = gpt35.run(prompt)
        generated_answer = generated_answer.strip()

        if num_questions % 500 == 0:
            print(f'--({i}) {num_correct}/{num_questions}')

        num_questions += 1
        if generated_answer in correct_answers:
            num_correct += 1

    accuracy = num_correct / num_questions
    print(f"({i}): {accuracy}")
    accuracies.append(accuracy)

    print("Final accuracy:", np.mean(accuracies))