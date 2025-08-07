import requests
import logging 
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Model


NEWS_API_URL = "https://newsapi.org/v2/top-headlines"
API_KEY = "fd5dd9059e6c423ab6d91ea447f2e72d"
HEADLINE_COUNT =3  
ARTICLE_COUNT = 5

logging.basicConfig(level=logging.INFO, format= "%(asctime)s - %(levelname)s -%(messgae)s")


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gen_model = GPT2LMHeadModel.from_pretrained("gpt2")
emb_model = GPT2Model.from_pretrained("gpt2")

# Add Padding Tokens {Becuase this is not default in GPT2}
tokenizer.pad_token = tokenizer.eos_token

gen_model.pad_token_id = gen_model.config.eos_token_id
emb_model.pad_token_id = emb_model.config.eos_token_id


def get_mean_embedding(input_text):
    input_text_tokens = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        embeddings = emb_model(**input_text_tokens)
    return embeddings.last_hidden_state.mean(dim=1) # Share [1,768] 768-d


def fetch_news_articles(api_key, count) -> list[dict]:
    logging.info ("Fetching new articles.... Please wait")
    news_api_response =  requests.get(NEWS_API_URL, params = {
        "apiKey" : api_key,
        "country" : "us",
        "pageSize" : count
    })

    news_api_response.raise_for_status()
    news_articles = news_api_response.json().get("articles", [])
    logging.info (f"Fetched {len(news_articles)} articles")
    return news_articles


def generate_headlines(prompt):
    news_headline_tokens = tokenizer(prompt, return_tensors="pt").input_ids
    headlines = gen_model.generate (
        news_headline_tokens,
        num_return_sequences = 1, # Fix the issue for Greedy - Beam message
        max_new_tokens =50,
        temperature = 0.8,
        top_k=50,
        top_p  = 0.95
    )
    return [tokenizer.decode(headline, skip_special_tokens = True).replace(prompt, "").strip() for headline in headlines]



def process_news_articles(news_articles):

    # Get Each News Article and get the Top 3 (count) Headlines
    for id, news_article in enumerate(news_articles):
        content = news_article.get("content") or ""
        title = news_article.get("title") or ""
        description = news_article.get("description") or ""

        # News Article =  "title + description + content"
        news_text = f"{title}. {description}. {content}".strip()

        if not news_text: # if Empty them get next article : continue
            continue

        # Prepare Prompt : Get Healine  for Each Article (You are in for loop)
        prompt = f"Generate a headline for this news: \n {news_text} \n Headline:"
        headline_candidates = generate_headlines(prompt) # For a Article

        # Get the Original Prompt Embeddings
        news_text_embedding = get_mean_embedding(prompt)

        result = []
        for headline in headline_candidates:
            headline_mean_embedding =  get_mean_embedding(headline)
            score = F.cosine_similarity(headline_mean_embedding,news_text_embedding).item()
            result.append((headline, score)
            )

        best_headline, score = max(result, key= lambda x: x[1])
        print (f"{best_headline} - Score {score:.4f}")

        ## Todo : Write to the JSON File with 
        ## Article :
        ##      Article
        ##          News_Text
        ##          Headline
        ##              Score


if __name__ == "__main__":
    #try:
        # Get the News Articles from the News API
        articles = fetch_news_articles(API_KEY, ARTICLE_COUNT) # Get top 5 Articles
        
        # Process Articles from New API
        process_news_articles (articles)

   # except Exception as e:
   #     logging.info (str(e))
    