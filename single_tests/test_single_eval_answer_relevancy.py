# from deepeval.metrics import AnswerRelevancyMetric
# from deepeval.test_case import LLMTestCase

# test_case = LLMTestCase(
#     input="What do penguins eat?",
#     actual_output="Penguins primarily feed on marine animals such as krill and tiny fish"
# )

# metric = AnswerRelevancyMetric(threshold=0.7)

# metric.measure(test_case)

# print("Score: ", metric.score)
# print("Reason:", metric.reason)
# print("Passed:", metric.is_successful())

import os

from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import GeminiModel

# Make sure GOOGLE_API_KEY is configured in your environment.
assert os.getenv("GOOGLE_API_KEY"), "GOOGLE_API_KEY is not set"

# Configure Gemini as the evaluation model
gemini_model = GeminiModel(
    model="gemini-3.1-flash-lite",
    api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0,
)

test_case = LLMTestCase(
    input="What do penguins eat?",
    actual_output=(
        "Penguins primarily feed on marine animals "
        "such as krill and tiny fish."
    ),
)

metric = AnswerRelevancyMetric(
    threshold=0.7,
    model=gemini_model,
    include_reason=True,
)

metric.measure(test_case)

print("Score:", metric.score)
print("Reason:", metric.reason)
print("Passed:", metric.is_successful())
