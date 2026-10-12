from RAG1.agent import ask
import json
import time

with open("./eval_dataset.jsonl", "r", encoding="utf-8") as file:
    test_cases = [json.loads(line) for line in file]
    
    results = []
    for test_case in test_cases:
        question = test_case["question"]
        excepted_answer = test_case["expected_answer"]

        print(f"\n Question: {question}")

        rag_result = ask(question)
        # rag_result = json.loads(rag_result)

        result = {
            "question": question,
            "expected_answer": excepted_answer,
            "actual_answer": rag_result["answer"],
            "context": rag_result["context"]
        }

        results.append(result)
        with open("results1.jsonl", "w", encoding="utf-8") as file:
            for result in results:
                file.write(json.dumps(result)+"\n")

        print(f" Answer: {rag_result['answer']}")
        print("============= Sleep ======")
        time.sleep(5)

with open("results1.jsonl", "w", encoding="utf-8") as file:
    for result in results:
        file.write(json.dumps(result)+"\n")

print("\n Done! Results1 saved to results1.jsonl")