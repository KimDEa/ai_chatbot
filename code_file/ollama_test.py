# import packages

import ollama
import json
import os
import pandas as pd
import numpy as np
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import time

load_dotenv()

class SearchQuery(BaseModel): # pydantic로 각 카테고리에 데이터 타입 규정.
    type: str | None = Field(
        description="Movie 또는 TV Show"
    )
    title: list[str] = Field(
        description="영화 제목. 없다면 빈리스트"
    )
    director: list[str] = Field(
        description="감독이름. 없다면 빈리스트"
    )
    cast: list[str] = Field(
        description="출연진 배우 이름. 없다면 빈리스트"
    )
    country: str | None = Field(
        description="만든 국가"
    )
    release_year: int | None = Field(
        description="연도가 하나만 언급되면 여기에 저장"
    )
    year_from: int | None = Field(
        description="시작 연도. 예: 2000's -> 2000, 'from 2000' -> 2000"
    )
    year_to: int | None = Field(
        description="끝 연도. 예: 2000's -> 2009, 'to 2009' -> 2009"
    )
    duration: str | None = Field(
        description="상영시간은 '90 min'식으로 저장"
    )
    listed_in: list[str] = Field(
        description="장르. 없다면 빈리스트"
    )

class ChatBot:
    def __init__(self): # 생성자 생성
        self.N_PATH = "/Users/d.h/dev/ai_chatbot/data_file/netflix_titles.csv"
        self.V_PATH = "/Users/d.h/dev/ai_chatbot/data_file/netflix_desctiption.npy"
        self.P_PATH = "/Users/d.h/dev/ai_chatbot/data_file/netflix_pd.json"
        self.api_key = os.environ["GEMINI_API_KEY"]

        self.MODEL = "bge-m3"
        self.BATCH = 100
    
    def dataframe(self): # 넷플릭스 csv파일 데이터프레임화, 특정 설명, show_id부분 리스트로 변환후 반환.
        df = pd.read_csv(self.N_PATH)
        df = df.fillna("Unknown")
        descs = df["description"].tolist()
        ids = df["show_id"].tolist()
        
        return df, descs, ids
    
    def call_gemini(self, user_input): # gemini api 호출해서 사용자 입력 카테고리에 맞게 분류.
        genres = sorted(
            df['listed_in'].str.split(',').explode().str.strip().unique()
        )
        #genres = [g for g in df['listed_in']]

        prompt = (
            f"""
            사용자가 보고싶은 영화나 TV프로그램을 자연어 방식으로 입력하면
            너는 {user_input}을 각 카테고리에 맞는 키워드를 넣어주는 역활이야.
            장르부분은 {genres}안에 포함된것만 넣어줘.
            """
        )

        client = genai.Client(api_key=self.api_key)

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
            print(f"API Call Failure: {e}")
            print("60초 동안 대기후 재요청.")
            time.sleep(61)
            for i in range(3):
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
                if keyword_result:
                    break
            return keyword_result

    def extract_pd(self, keyword_result): # 각 카테고리에 있는 키워드들을 pandas로 추출후 json형식으로 파일에 저장.
        df_c = df.copy()

        if keyword_result['type']:
            df_c = df_c[df_c['type'] == keyword_result['type']]
        if keyword_result['title']:
            t = "|".join(keyword_result['title'])
            df_c = df_c[df_c['title'].str.contains(t, case=False, na=False)]
        if keyword_result['director']:
            d = "|".join(keyword_result['director'])
            df_c = df_c[df_c['director'].str.contains(d, case=False, na=False)]
        if keyword_result['cast']:
            c = "|".join(keyword_result["cast"])
            df_c = df_c[df_c['cast'].str.contains(c, case=False, na=False)]
        if keyword_result['country']:
            df_c = df_c[df_c['country'].str.contains(keyword_result['country'], case=False, na=False)]
        if keyword_result['release_year'] and not keyword_result['year_from'] or not keyword_result['year_to']:
            df_c = df_c[df_c['release_year'] == keyword_result['release_year']]
        if keyword_result['year_from'] and keyword_result['year_to']:
            df_c = df_c[df_c['release_year'].between(keyword_result['year_from'], keyword_result['year_to'])]
        if keyword_result['duration']:
            df_c = df_c[df_c['duration'].str.contains(keyword_result['duration'], case=False, na=False)]
        if keyword_result['listed_in']:
            l = "|".join(keyword_result['listed_in'])
            df_c = df_c[df_c['listed_in'].str.contains(l, case=False, na=False)]
        
        results = df_c.to_dict(orient="records")
        with open(self.P_PATH, mode="w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
    def input_embed(self, user_input):
        # 사용자 입력 벡터화.
        result = ollama.embed(model=self.MODEL, input=user_input)
        input_array = np.array(result["embeddings"][0], dtype=np.float32)
        input_length = input_array / np.linalg.norm(input_array)

        if np.allclose(np.linalg.norm(input_length), 1) == True:
            print(f"사용자 입력 데이터 벡터화 완료: {input_length.shape} / 데이터 길이값 테스트 통과.")
            return input_length
        else:
            return print("사용자 입력 데이터 벡터화 오류. 다시 시도하세요.")

    def desc_embed(self, descs):
        # 설명부분 벡터화.
        vectors = []

        for i in range(0, len(descs), self.BATCH):
            batch = descs[i:i + self.BATCH]
            result = ollama.embed(model=self.MODEL, input=batch)
            vectors.extend(result["embeddings"])
            if len(batch) < len(descs):
                print(f"임베딩중... {len(vectors)} / {len(descs)}")

        desc_array = np.array(vectors, dtype=np.float32)
        desc_length = desc_array / np.linalg.norm(desc_array, axis=1, keepdims=True)

        if len(descs) == len(vectors) and np.allclose(np.linalg.norm(desc_length, axis=1, keepdims=True), 1) == True:
            print(f"설명 데이터 벡터화 완료: {desc_length.shape} / 데이터 길이값 테스트 통과.")
            return desc_length
        else:
            return print("설명 데이터 벡터화 오류. 다시 시도하세요.")

    def compare_vectors(self, input_length, desc_length):
        pass
        

# 사용자 입력, 영화 설명부분 ollama로 임베딩후 행 길이 맞추기

# 두 벡터 비교후 상위 N개 추출

# show_id로 두 추출 데이터 교집합 데이터 추출

# gemini api로 ai스럽게 답변

if __name__ == "__main__":
    while True: # 사용자 입력
        user_input = str(input("user : "))
        if user_input in ["q", "quit", "exit"]:
            print("quit the program...")
            break
        else:
            c = ChatBot()
            df, descs, ids = c.dataframe()
            keyword_result = c.call_gemini(user_input)
            c.extract_pd(keyword_result)
            input_length = c.input_embed(user_input)
            desc_length = c.desc_embed(descs)
