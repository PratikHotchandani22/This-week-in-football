REDDIT_SUBREDDITS = ["soccer","championsleague","football","seriea","PremierLeague","LaLiga","Bundesliga","Ligue1"]

tgt_lang = "en"
BATCH_SIZE = 100
REDDIT_COMMENT_CLEANING_LABELS = ['Bot','Human-Conversation','N/A']
REDDIT_COMMENT_CLEANING_LABELS_STR = ",".join(REDDIT_COMMENT_CLEANING_LABELS)

MODEL_COMMENT_CLEANING = "llama3.1"
MODEL_COMMENT_CLEANING_2 = "llama3.2:1b"

PROMPT_COMMENT_CLEANING = {
    "name": "prompt_a",
    "content": (
        "You are a Reddit subreddit moderator whose task is to categorize comments into one of the following labels. "
        "You must output **exactly one** of these labels, whichever is the most likely. You cannot output anything other than one of these labels. "
        "If you're unsure, output **only N/A**.\n\n"
        f"Labels: {REDDIT_COMMENT_CLEANING_LABELS_STR}.\n\n"
        "Interpretation of some labels:\n\t"
        "Bot: A comment that appears to have been generated by a bot.\n\t"
        "Human-Conversation: A comment that reflects human input, offering opinions, reactions, or conversational responses with substance.\n\t"
        "N/A: Use this when a comment does not contribute insight to the conversation or context, such as short, vague statements or irrelevant replies.\n\n"
        "Output format: **one label**.\n\n"
        'Example 1: "I am a bot, and this action was performed automatically. Please [contact the moderators of this subreddit](/message/compose/?to=/r/PremierLeague) if you have any questions or concerns." -> "Bot".\n\n'
        'Example 3: "After the game, you can now legitimately say that Company is doing a really good job so far." -> "Human-Conversation".\n\n'
        'Example 4: "The worst building I’ve seen with us for a long time. High and away." -> "Human-Conversation".\n\n'
        'Example 5: "No comments." -> "N/A".\n\n'
        "Please output only **one label** from the list: `Bot`, `Human-Conversation`, or `N/A`. No other text should be included."
        "\n\nThe comment text to categorize is provided inside ```.\n\n"
    ),
}

