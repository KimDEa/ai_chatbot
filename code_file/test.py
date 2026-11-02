import os
import numpy as np
import ollama
import pandas as pd
from google import genai
import json
import re


class Chatbot:
    def __init__(self):
        self.MODEL = "bge-m3"
        self.BATCH = 100
        self.N_PATH = "/Users/d.h/dev/ai_chatbot/data_file/netflix_titles.csv"
        self.B_PATH = "/Users/d.h/dev/ai_chatbot/data_file/netflix_desctiption.npy"
        self.STOPWORDS = {"i", "a", "an", "the", "to", "want", "watch", "show", "me",
             "in", "on", "of", "and", "or", "some", "something", "about",
             "released", "from", "with", "for", "is", "are", "any"}
        self.search_columns = ["type", "listed_in", "country", "release_year"]

    def dataframe(self):
        df = pd.read_csv(self.N_PATH)
        df = df.fillna("Unknown")
        descs = df["description"].tolist()
        show_ids = df["show_id"].tolist()

        #print(df["show_id"].shape)
        #print(df["description"].shape)
        return df, descs, show_ids

    def embed(self, descs):
        vectors = []
        
        for i in range(0, len(descs), self.BATCH):
            batch = descs[i:i + self.BATCH]
            result = ollama.embed(model=self.MODEL, input=batch)
            vectors.extend(result["embeddings"])
            if len(descs) > self.BATCH:
                print(f"임베딩 {len(vectors)}/{len(descs)}")
        vectors = np.array(vectors, dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def load_or_write(self, descs, df):
        if os.path.exists(self.B_PATH):
            saved = np.load(self.B_PATH)
            if len(saved) == len(descs):
                print(f"변동 없는 npy파일 존재함. {df["description"].shape} / {saved.shape}")
            else:
                print(f"파일 내용이 다름. npy파일 업데이트 필요. {df["description"].shape} / {saved.shape}")
                vectors = self.embed(descs)
                np.save(self.B_PATH, vectors)
        else:
            print("벡터파일이 존재하지 않음. 새로 만드는중...")
            vectors = self.embed(descs)
            np.save(self.B_PATH, vectors)

    def split_by_input(self, user_input):
        words = user_input.split()
        re_stopword = [w for w in words if w not in self.STOPWORDS]
        return re_stopword

    def split_by_pandas(self, df, re_stopword):
        for word in re_stopword:
            mask = df[self.search_columns].apply(
                lambda col: col.astype(str).str.lower().str.contains(word, na=False)
            ).any(axis=1)

        results = df[mask]
        data = results.to_dict(orient="records")
        with open("/Users/d.h/dev/ai_chatbot/data_file/netflix_pd.json", mode="w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)


class Poo:
    def test(self):
        df = pd.read_csv("/Users/d.h/dev/ai_chatbot/data_file/netflix_titles.csv")
        df = df.fillna("Unknown")

        descs = df["description"].tolist()

        input = "i want to watch horror movie released in 2012"
        words = input.split()

        print(words)
        

if __name__ == "__main__":
    #p = Poo()
    #p.test()
    c = Chatbot()
    df ,descs, show_ids = c.dataframe()
    c.load_or_write(descs, df)
    
    while True:
        user_input = str(input("user👤: ")).lower()
        if user_input in ("q", "quit", "exit"):
            print("Exiting the Program.")
            break
        else:
            re_stopword = c.split_by_input(user_input)
            c.split_by_pandas(df, re_stopword)