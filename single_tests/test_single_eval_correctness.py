
import os

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams
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
    input="Where do penguins live?",
    actual_output="Penguins live in the Southern Hemisphere, inhabiting a variety of environments ranging from the icy coasts of Antarctica and the cool shores of South America and southern Africa to the Galapagos Islands near the equator.",
    expected_output="Penguins live in the Southern Hemisphere, including Antarctica, South America, southern Africa, and the Galapagos Islands."
)

metric = GEval(
    name="Correctness",
    criteria=(
        "Determine whether the actual output is factually correct based on the expect output"
    ),
    evaluation_params=[
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
    ],
    # threshold=0.7,
    model=gemini_model,
    # include_reason=True,
)

metric.measure(test_case)

print("Score:", metric.score)
print("Reason:", metric.reason)
