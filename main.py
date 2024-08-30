from sentence_transformers import SentenceTransformer, util
import warnings
import pyperclip

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

model = SentenceTransformer('all-MiniLM-L6-v2')

questions = [
    "How can I find my scrapbook link?",

]
answers = [
    "To find your scrapbook link find your scrapbook post in #scrapbook channel, right click on the post and click on 'Copy link'. \n\nIf you have trouble finding it, use the search filters to search for a message sent by you in the #scrapbook channel.",

]

question_embeddings = model.encode(questions, convert_to_tensor=True)

def get_best_answer(user_query):
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    
    similarities = util.pytorch_cos_sim(query_embedding, question_embeddings)[0]
    
    best_match_idx = similarities.argmax()
    
    threshold = 0.7
    if similarities[best_match_idx] >= threshold:
        return answers[best_match_idx]
    else:
        return "I'm not sure how to answer that, could you clarify?"


while True:
    user_query = input("Ask me a question: ")
    answer = get_best_answer(user_query)
    pyperclip.copy(answer)
    print(answer)