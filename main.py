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
    "On average how much time does it take for an iPad order to be processed?",
    "How do I know if I went into the next round in showcase?",
    "Does anybody know how long the >70 ticket interview lasts?",
    "How long before a zoom call is setup for a big purchase?",
    "Is there any way to check if all my scrapbooks have been reviewed?",
    "Usually how long does it take for an arcade order to get processed?",
    "Why can’t I open /shop?",
    "How long can it take to review a project?",
    "Where can I see my public page of the showcase?",
    "Where do I see if I advanced to the next round?",
    "Is the bot down?",
    "Can someone check my screenshot for errors?",
    "How do I show a live demo of my artwork?",
    "How many artworks can I put in for the showcase?",
    "How long is the average wait-time for the iPad zoom call?",
    "Is there still time to buy something from the shop after a session review?",
    "How do I check the status of a session?",
    "How can I find my scrapbook link?",
    "How do I check if my session's time was recorded correctly?",
    "Is the bot down and what should I do if it is?",
    "Can I still scrapbook my hours if the game isn’t 100% working?",
    "What should I do if my progress timer got reset?",
    "How do I handle an error with linking sessions to a scrapbook post?",
    "Can I submit arcade projects after the deadline?",
    "What’s the status of a specific order with a tracking number?",
    "How do I find out the backlog of scrapbooks?",
    "What if I can’t link multiple sessions to a scrapbook post?",
    "How can I find out if I moved to the next round in showcase?",
    "Does Hack Club pay VAT for orders?",
    "How do I attach sessions to a scrapbook post without a DM from Hakkuun?",
    "How can I check the reviewing progress of my project?",
    "What’s the difference between 'Approved' and 'Banked' sessions?",
    "How long before a zoom call is set up for a big purchase?",
    "Do I have to attend another zoom call if I've already had one before for a purchase?",
    "What should I ensure in my projects to make the review process smooth?",
    "Can I get credits from other members to buy a 3D printer?",
    "Is there a list of all Hack Club programs?",
    "What to do if I receive an error while submitting votes for showcase?",
    "How do I check the status of digital ocean credits after ordering?",
    "Can someone help me review my sessions?",
    "What should I do if my scrapbook link is not working?"
]



answers = [
    "To find your scrapbook link, locate your scrapbook post in the #scrapbook channel, right-click on the post, and select 'Copy link.' If you're having trouble, use search filters to look for messages sent by you in the #scrapbook channel.",
    "If you've already attended a Zoom call for an item you've ordered, you don't need to attend another one. You'll receive a DM to confirm your details instead.",
    "Reviews are not available on request. Please be patient and wait for your turn. Reviewers will review sessions in the order they are received.",
    "To make the review process smooth, ensure your projects include clear documentation, a concise README, and a well-organized repository. Check out this guide for more details: https://github.com/hackclub/arcade-constitution",
    "Processing time for an iPad order can vary. On average, it takes between a few days to a few weeks. Please be patient and wait for a DM regarding your order status.",
    "To find out if you advanced to the next round in the showcase, check the #arcade-showcase-help channel or any updates provided by the showcase team.",
    "The >70 ticket interview typically lasts between 5 to 15 minutes. The interviewer will confirm your details and ask questions about your projects.",
    "A Zoom call for a big purchase is usually set up within a few days after a successful re-review. Please be patient and wait for a DM with the details.",
    "You can check if all your scrapbooks have been reviewed by running the /shop command and checking the status of your sessions.",
    "For smaller arcade orders, processing usually takes 1-2 days. For larger orders, it may take up to 1.5 weeks. Please wait for a DM with your order status.",
    "If you can't open /shop, try reloading your Slack or closing and reopening the app. If the issue persists, check for any announcements or status updates from the Hack Club team.",
    "The time to review a project can vary. Generally, it depends on the backlog and complexity of the project. Check with the review team for more specific information.",
    "You can see your public page of the showcase by checking your showcase post or following any links provided in the showcase channel or updates.",
    "To see if you advanced to the next round, check the showcase announcements or reach out in the #arcade-showcase-help channel.",
    "If the bot is down, check the status page at https://status.hackclub.com/?duration=1d for updates and wait for the bot to come back online.",
    "For issues with screenshots or errors, post your screenshots in the relevant channel or thread and ask for help from the community or review team.",
    "To show a live demo of your artwork, you can use screen-sharing tools during a Zoom call or record a video of your artwork in action and share the link.",
    "The number of artworks you can showcase may vary. Check with the showcase guidelines or ask in the #arcade-showcase-help channel for specific limits.",
    "The average wait-time for an iPad Zoom call is usually a few days after order placement or review. Check for updates in your DMs or contact support for details.",
    "After a session review, you should still have time to buy items from the shop unless otherwise stated. Check the shop for availability and deadlines.",
    "To check the status of a session, use the /shop command or check the status update in your session thread.",
    "To find your scrapbook link, use the same method as before: locate the post in the #scrapbook channel, right-click, and copy the link.",
    "If you suspect the progress timer was not recorded correctly, post in the relevant channel with details of the issue and ask for assistance.",
    "If the bot is down, post your session updates manually and inform the community about the issue. Wait for the bot to come back online to update your progress.",
    "If your game isn’t fully working, you can still scrapbook your hours, but make sure to clearly indicate any issues and progress made. This will help in the review process.",
    "If your progress timer got reset due to a bot issue, post in the relevant channel with details and seek assistance. Wait for the bot to come back online for proper tracking.",
    "If you encounter an error linking sessions to a scrapbook post, report the issue in the relevant channel with details and screenshots to get support from the community or team.",
    "Yes, you can still submit arcade projects after the deadline, but check if there are any specific rules or extensions provided by the Hack Club team.",
    "To check the status of a specific order, use the tracking number on the carrier's website or contact Hack Club support for updates.",
    "The backlog of scrapbooks can be checked by asking in the relevant channel or checking any updates provided by the review team.",
    "If you can’t link multiple sessions to a scrapbook post, try troubleshooting the issue by refreshing or restarting the app. If the problem persists, seek help from the community.",
    "To find out if you moved to the next round in the showcase, check the showcase announcements or reach out in the #arcade-showcase-help channel.",
    "Hack Club typically does not cover VAT for orders. Check the specific terms and conditions for your order or contact support for details.",
    "If you didn't get a DM from Hakkuun, manually attach your sessions to your scrapbook post using the provided instructions or contact support for assistance.",
    "To check the reviewing progress of your project, monitor any updates in the relevant channel or reach out to the review team for current status.",
    "An 'Approved' session means it has been reviewed and accepted. A 'Banked' session is pending review. Only 'Approved' sessions count for credits.",
    "A Zoom call for a big purchase is typically set up within a few days after the review. Check for updates in your DMs or contact support for specific timing.",
    "You generally do not need to attend another Zoom call if you've already had one for a previous purchase. However, check with Hack Club for any specific requirements.",
    "To ensure a smooth review process, include clear documentation, a well-organized repository, and follow any specific guidelines provided by Hack Club.",
    "You cannot transfer credits between members. Each member must use their own credits for purchases. Consider collaborating on a shared project to earn more credits.",
    "A list of all Hack Club programs can be found on their website or by contacting support for detailed information about available programs and initiatives.",
    "If you encounter an error while submitting votes for the showcase, report the issue and try again later. Contact support if the problem persists.",
    "If you haven’t received your digital ocean credits, check your email for any confirmation and contact support if there’s a delay beyond the expected timeframe.",
    "To get help with reviewing your sessions, post in the relevant channel or contact the review team with details of your sessions and any issues.",
    "If your scrapbook link is not working, check if it’s the correct link and ensure it’s publicly accessible. Seek help from the community if issues persist."
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