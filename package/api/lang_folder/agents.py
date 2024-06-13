from lang_folder.few_shot_examples import few_shot_examples
from lang_folder.chains import classification_chain
# Function to classify a given input string
def classify_input_string(input_string):
    # Prepare the input for the LLMChain
    input_data = {"input_string": input_string}
    print(f"Dataa to classification chain {input_data}")
    # Invoke the LLMChain to get the classification
    result = classification_chain.invoke(input_data)
    return result['text']


# This will later be converted to an API call
async def get_few_shot_examples():
    return few_shot_examples
