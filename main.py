from sentence_transformers import SentenceTransformer, util
import warnings
import pyperclip

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

model = SentenceTransformer('all-MiniLM-L6-v2')

questions = [
    "How can I find my scrapbook link?",
    "Do I have to attend another zoom meeting/call if I have ordered an item from the store before?",
    "Can someone review my sessions?",
    "What are some things I should ensure are present in my projects to make the review process smooth, fast, and easy for reviewers?",
    "On average how much time does it take for an ipad order to be processed?",
    "How do I know if I went into the next round in showcase?",
    "Does anybody know how long the >70 ticket interview lasts?",
    "How long before a zoom call is setup for a big purchase?",
    "Is there any way to check if all my scrapbooks have been reviewed?",
    "Usually how long does it take for an arcade order to get processed?",
    "Why can’t i open /shop?"

]
answers = [
    "To find your scrapbook link find your scrapbook post in #scrapbook channel, right click on the post and click on 'Copy link'. \n\nIf you have trouble finding it, use the search filters to search for a message sent by you in the #scrapbook channel.",
    "If you already attended a zoom call for an item you ordered, you don't have to attend another one. You'll get a DM to confirm your details instead.",
    "No reviews on request. Please be patient and wait for your turn.",
    "https://github.com/hackclub/arcade-constitution",
    "It can take a few days to a few weeks for your order to be processed. Please be patient and wait for a DM!",
    "Please ask in the #arcade-showcase-help channel.",
    "The interview lasts for 5-15 minutes. They confirm your details and ask you a few questions about your projects.",
    "A zoom call is usually setup within a few days after the successful re-review. Please be patient and wait for a DM!",
    "You can check if your scrapbooks have been reviewed by running /shop and checking the status of your sessions.",
    "For smaller orders it can take max 1-2 days. For larger orders it will take max 1.5 weeks. Please be patient and wait for a DM!",
    "Reload your Slack and try again! If you're on mobile try closing and reopening the app."
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