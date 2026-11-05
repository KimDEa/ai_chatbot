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
import time

load_dotenv()


class SearchQuery(BaseModel):
    type: str | None = Field(
        description='"Movie" 또는 "TV Show". 언급이 없으면 Null'
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

    def dataframe(self): # 넷플릭스 데이터 프레임 생성후 'description', 'show_id'부분 리스트로 추출.
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
                print(f"임베딩... {len(vectors)}/{len(descs)}")
        vec_array = np.array(vectors, dtype=np.float32)
        vec_length = vec_array / np.linalg.norm(vec_array, axis=1, keepdims=True)
        return vec_length

    def load_or_write(self, descs, df): # npy파일이 없으면 다시 만들고 저장, 있으면 그대로 불러오기.
        if os.path.exists(self.B_PATH):
            saved = np.load(self.B_PATH)
            norm_test = np.allclose(np.linalg.norm(saved, axis=1, keepdims=True), 1)
            if len(saved) == len(descs) and norm_test == True:
                print(f"npy파일 무결성 테스트 통과. {df["description"].shape} / {saved.shape}")
                return 0
            else:
                print(f"파일 내용이 다르거나 벡터의 길이가 일정하지 않음. npy파일 업데이트 필요. {df["description"].shape} / {saved.shape}")
                vectors = self.embed(descs)
                np.save(self.B_PATH, vectors)
        else:
            print("벡터파일이 존재하지 않음. 새로 만드는중...")
            vectors = self.embed(descs)
            norm_test = np.allclose(np.linalg.norm(vectors, axis=1, keepdims=True), 1)
            if norm_test is True:
                print(f"데이터 무결성 테스트 통과. npy파일로 저장합니다.")
                np.save(self.B_PATH, vectors)

    def split_by_input(self, user_input): # 입력을 받아서 불필요한 i, am 같은 문자 없애고 리스트로 반환.
        words = user_input.split()
        re_stopword = [w for w in words if w not in self.STOPWORDS]
        print(re_stopword)
        return re_stopword

    def extract_input(self, df, user_input): # 사용자 입력을 Gemini api를 이용해 알맞은 키워드를 분류. 
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
            return keyword_result

        except Exception as e:
            time.sleep(61)
            print(f"api 요청 실패: {e}")
            
            keyword_result = self.extract_input(df, user_input)
            return keyword_result

    def extract_from_pd(self, df, keyword_result): # Gemini api로 분류받은 키워드를 pandas로 데이터 추출.
        result = df.copy()

        if keyword_result['type']:
            result = result[result['type'] == keyword_result['type']]
        if keyword_result['title']:
            t = "|".join(keyword_result['title'])
            result = result[result['title'].str.contains(t, na=False, case=False)]
        if keyword_result['director']:
            d = "|".join(keyword_result['director'])
            result = result[result['director'].str.contains(d, na=False, case=False)]
        if keyword_result['cast']:
            c = "|".join(keyword_result['cast'])
            result = result[result['cast'].str.contains(c, na=False, case=False)]
        if keyword_result['country']:
            c = "|".join(keyword_result['country'])
            result = result[result['country'].str.contains(c, na=False, case=False)]
        if keyword_result['listed_in']:
            l = "|".join(keyword_result['listed_in']) 
            result = result[result['listed_in'].str.contains(l, na=False, case=False)]
        if keyword_result['release_year']:
            result = result[result['release_year'] == keyword_result['release_year']]
        if keyword_result['year_from'] and keyword_result['year_to']:
            result = result[result['release_year'].between(keyword_result['year_from'], keyword_result['year_to'])]

        result_json = result.to_dict(orient="records")
        with open("/Users/d.h/dev/ai_chatbot/data_file/netflix_pd.json", mode="w", encoding="utf-8") as f:
            json.dump(result_json, f, indent=4, ensure_ascii=False)

        return result_json

    def compare_vectors(self, df, user_input): # 입력 벡터랑 설명 벡터 비교후 넷플릭스 데이터 json으로 저장.
        input_embed = ollama.embed(model=self.MODEL, input=user_input)
        input_vec = input_embed["embeddings"][0]
        input_array = np.array(input_vec, dtype=np.float32)
        input_length = input_array / np.linalg.norm(input_array)
        input_test = np.allclose(np.linalg.norm(input_length), 1)

        desc_vec = np.load(self.B_PATH)
        desc_test = np.allclose(np.linalg.norm(desc_vec, axis=1, keepdims=True), 1)

        if input_test is True and desc_test is True:
            print("사용자 입력 벡터와 npy파일의 벡터의 무결성 테스트 완료. 둘의 유사도 비교를 시작합니다.")
            compare_result = desc_vec @ input_length

            print(f"결과: {compare_result.shape}")
            print(compare_result[:5])
            return compare_result
        else:
            print("사용자 입력 벡터와 npy파일의 벡터의 무결성 테스트 실패. 다시 시도하세요.")
            return
    
    def extract_from_vec(self, df, compare_result):
        vec_result = [df.iloc[i].to_dict() for i in range(len(compare_result)) if compare_result[i] > 0.5]
        
        with open("/Users/d.h/dev/ai_chatbot/data_file/netflix_vec.json", mode="w", encoding="utf-8") as f:
            json.dump(vec_result, f, indent=4, ensure_ascii=False)
        
        return 

    def norm_test(self, df, descs):
        vectors = []

        desc_list = [df["description"][i] for i in range(200)]
        

        for i in range(0, len(desc_list), self.BATCH):
            batch = desc_list[i:i + self.BATCH]
            results = ollama.embed(model=self.MODEL, input=batch)
            vectors.extend(results["embeddings"])
            if len(desc_list) > len(vectors):
                print(f"임베딩... {len(vectors)}/{len(descs)}.")

        print(f"임베딩후 vectors의 데이터 타입: {type(vectors)}")
        #print(f"임베딩후 vectors의 shape: {vectors.shape}")

        vec_array = np.array(vectors, dtype=np.float32)
        print(f"배열후 vectors의 데이터 타입: {type(vec_array)}")
        print(f"배열후 vectors의 shape: {vec_array.shape}")

        vec_norm = np.linalg.norm(vec_array, axis=1, keepdims=True)
        print(f"norm 이후 vec_array의 데이터 타입: {type(vec_norm)}")
        print(f"norm 이후 vec_array의 shape: {vec_norm.shape}")

        vec_length = vec_array / vec_norm
        print(f"배열의 길이값을 나눈후의 데이터 타입: {type(vec_length)}")
        print(f"배열의 길이값을 나눈후의 shape: {vec_length.shape}")

        is_one = np.allclose(np.linalg.norm(vec_length, axis=1, keepdims=True), 1)
        if is_one is True:
            print("데이터 무결성 테스트 통과")
        else:
            print("데이터 무결성 테스트 실패")
        
        
        
        
                

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

        horror_genres = [
            g.strip()
            for row in df["listed_in"].dropna()
            for g in row.split(",")
            if "Horror" in g
        ]
        print(horror_genres)


if __name__ == "__main__":
    c = Chatbot()
    df ,descs, show_ids = c.dataframe()
    c.load_or_write(descs, df)

    while True:
        user_input = str(input("user👤: ")).lower()
        if user_input in ("q", "quit", "exit"):
            print("Exiting the Program.")
            break
        else:
            keyword_result = c.extract_input(df, user_input)
            c.extract_from_pd(df, keyword_result)
            compare_result = c.compare_vectors(df, user_input)
            c.extract_from_vec(df, compare_result)