import os
import numpy as np
import ollama
import pandas as pd
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import json
import re

load_dotenv()


class SearchQuery(BaseModel):
    type: str | None = Field(
        description='"Movie" 또는 "TV Show". 언급이 없으면 빈 리스트'
    )
    title: list[str] = Field(
        description='영화제목. 언급이 없으면 빈 리스트'
    )
    director: list[str] = Field(
        description='감독이름. 언급이 없으면 빈 리스트'
    )
    cast: list[str] = Field(
        description='출연진 이름. 여러개일수 있는데 언급이 없으면 빈 리스트'
    )
    country: list[str] = Field(
        description='만든 국가명. 언급이 없으면 빈 리스트'
    )
    listed_in: list[str] = Field(
        description=f"장르 목록. 언급이 없으면 빈 리스트"
    )
    year_from: int | None = Field(
        description="시작 연도. 예: 2000's → 2000, 'from 2000' → 2000"
    )
    year_to: int | None = Field(
        description="끝 연도. 예: 2000's → 2009, 'to 2012' → 2012"
    )
    release_year: int | None = Field(
        description="연도 하나만 언급되면 int형으로 저장. 언급이 없으면 null"
    )


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
        self.api_key = os.environ["GEMINI_API_KEY"]

    def dataframe(self): # 넷플릭스 데이터 프레임 생성
        df = pd.read_csv(self.N_PATH)
        df = df.fillna("Unknown")
        descs = df["description"].tolist()
        show_ids = df["show_id"].tolist()

        #print(df["show_id"].shape)
        #print(df["description"].shape)
        return df, descs, show_ids

    def embed(self, descs): # 영화 설명부분 벡터화후 npy로 저장.
        vectors = []
        
        for i in range(0, len(descs), self.BATCH):
            batch = descs[i:i + self.BATCH]
            result = ollama.embed(model=self.MODEL, input=batch)
            vectors.extend(result["embeddings"])
            if len(descs) > self.BATCH:
                print(f"임베딩 {len(vectors)}/{len(descs)}")
        vectors = np.array(vectors, dtype=np.float32)
        vectors = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors

    def load_or_write(self, descs, df): # npy파일이 없으면 다시 만들고 저장, 있으면 그대로 불러오기.
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

    def split_by_input(self, user_input): # 입력을 받아서 불필요한 i, am 같은 문자 없애고 리스트로 반환.
        words = user_input.split()
        re_stopword = [w for w in words if w not in self.STOPWORDS]
        print(re_stopword)
        return re_stopword

    def split_by_pandas(self, df, re_stopword): # pandas로만 분류.(Gemini api로 분류예정.)
        results = []
        for word in re_stopword:
            mask = df[self.search_columns].apply(
                lambda col: col.astype(str).str.lower().str.contains(word, na=False)
            ).any(axis=1)
            
            data = df[mask].to_dict(orient="records")
            results.extend(data)
        #data = results.to_dict(orient="records")
        with open("/Users/d.h/dev/ai_chatbot/data_file/netflix_pd.json", mode="w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    def extract_input(self, df, user_input):
        try:
            client = genai.Client(api_key=self.api_key)
        except Exception as e:
            print(e)
            return

        genres = sorted(
            df["listed_in"].str.split(',').explode().str.strip().unique()
        )

        prompt = (
            f"""
            사용자가 보고싶은 영화나 TV프로그램을 자연어 입력방식으로 하면
            너는 {user_input}을 받아서 {self.N_PATH}과 {SearchQuery}로 알맞게 분류를 해서 알맞은 데이터를 뽑을거야.
            검색은 {self.N_PATH}에 있는 데이터안에 있는것만 써야해.
            {genres}안에 포함된걸로만 장르로 구별해줘.
            """
        )

        try:
            response = client.models.generate_content(
                model='gemini-3.5-flash-lite',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0,
                    response_schema=SearchQuery,
                ),
            )
            keyword_result = json.loads(response.text)
            print(keyword_result)

        except Exception as e:
            print(e)

    def compare_vectors(self, df, user_input): # 입력 벡터랑 설명 벡터 비교후 넷플릭스 데이터 json으로 저장.
        result = ollama.embed(model=self.MODEL, input=user_input)
        input_vec = np.array(result["embeddings"][0], dtype=np.float32)
        #q = np.linalg.norm(input_vec)
        #print(input_vec)
        #print(input_vec.shape)

        desc_vec = np.load(self.B_PATH)
        normalize_list = []
        result_list = []
        

        for row in desc_vec:
            normalize_list.append(input_vec @ row)

        for i in range(len(normalize_list)):
            if normalize_list[i] >= 0.6:
                print(df.iloc[i, 2])
                result_list.append(df.iloc[i].to_dict())

        with open("/Users/d.h/dev/ai_chatbot/data_file/netflix_pd.json", mode="w", encoding="utf-8") as f:
            json.dump(result_list, f, indent=4, ensure_ascii=False)
                

class Poo:
    def test(self):
        user_input = ["I love horror movies", "I hate horror movies"]

        result = ollama.embed(model="bge-m3", input=user_input)
        a, b = np.array(result["embeddings"], dtype=np.float32)
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        #print(f"{float(a @ b):.3f}")
        print(a @ b)

    def compare_desc(self):
        df = pd.read_csv("/Users/d.h/dev/ai_chatbot/data_file/netflix_titles.csv")
        df = df.fillna("Unknown")

        print(df.iloc[0, 2])

        
        

if __name__ == "__main__":
    #p = Poo()
    #p.compare_desc()
    c = Chatbot()
    df ,descs, show_ids = c.dataframe()
    c.load_or_write(descs, df)

    while True:
        user_input = str(input("user👤: ")).lower()
        if user_input in ("q", "quit", "exit"):
            print("Exiting the Program.")
            break
        else:
            #re_stopword = c.split_by_input(user_input)
            #c.split_by_pandas(df, re_stopword)
            c.extract_input(df, user_input)
            #c.compare_vectors(df, user_input)