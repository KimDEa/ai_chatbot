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
from pathlib import Path

load_dotenv()

class SearchQuery(BaseModel): # pydantic으로 각 프로그램 카테고리에 따른 데이터 타입 규정.
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
    listed_in: list[str] = Field(
        description="장르. 없다면 빈리스트"
    )


class ChatBot:
    NETFLIX_RAW_PATH = "/Users/d.h/dev/ai_chatbot/data_file/netflix_titles.csv"
    DESC_NPY_PATH = "/Users/d.h/dev/ai_chatbot/data_file/netflix_desctiption.npy"
    NORM_PATH = "/Users/d.h/dev/ai_chatbot/data_file/netflix_vec.json"

    API_KEY = os.environ["GEMINI_API_KEY"]
    MODEL = "bge-m3"
    BATCH = 100

    def __init__(self):
        self.df = pd.read_csv(ChatBot.NETFLIX_RAW_PATH)
        self.df = self.df.fillna("Unknown")
        self.df_desc = self.df['description'].tolist()
        self.ids = self.df["show_id"].tolist()

        self.genres = sorted(
            self.df['listed_in'].str.split(',').explode().str.strip().unique()
        )

        self.client = genai.Client(api_key=ChatBot.API_KEY)

    def call_gemini(self, user_input): # gemini api 호출로 사용자 입력 문자열에서 알맞은 키워드 추출 및 분류.
        prompt = (
            f"""
            사용자가 찾고싶은 영화나 TV프로그램을 자연어 방식으로 입력하면
            너는 {user_input}을 각 카테고리에 맞는 키워드를 넣어주는 역활이야.
            장르부분은 {self.genres}안에 있는 문자열으로만 넣어줘.
            """
        )

        try:
            response = self.client.models.generate_content(
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
            print("60초 대기후 재시도...")
            time.sleep(61)
            for i in range(3):
                try:
                    response = self.client.models.generate_content(
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
                        return keyword_result
                except Exception as e:
                    print(f"API Call Failure: {e}")
                    print("60초 대기후 재시도...")
                    time.sleep(61)
                    continue
                else:
                    print("api 요청 성공.")

    def extract_pd(self, keyword_result): # gemini api로 분류된 키워드를 pandas 인덱싱으로 데이터 추출후 json형식으로 파일에 저장.
        df_c = self.df.copy()

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
        if keyword_result['listed_in']:
            l = "|".join(keyword_result['listed_in'])
            df_c = df_c[df_c['listed_in'].str.contains(l, case=False, na=False)]

        filter_results = df_c.to_dict(orient="records")

        #print(filter_results)
        return filter_results

    def input_embed(self, user_input):
        result = ollama.embed(model=ChatBot.MODEL, input=user_input)
        input_array = np.array(result["embeddings"][0], dtype=np.float32)
        input_norm = input_array / np.linalg.norm(input_array)

        if np.allclose(np.linalg.norm(input_norm), 1) == True:
            print(f"사용자 입력 데이터 정규화 완료: {input_norm.shape}")
            return input_norm
        else:
            retry = input("사용자 입력 데이터 정규화 오류. 다시 시도 하시겠습니까?(y/n) : ")
            if retry == "y":
                return c.input_embed(user_input)
            else:
                return

    def save_or_load(self): # 줄거리 부분 정규화 데이터 존재 유무 및 데이터 무결성 확인후 줄거리.npy 생성.
        if os.path.exists(ChatBot.DESC_NPY_PATH):
            saved = np.load(ChatBot.DESC_NPY_PATH)
            print("저장된 정규화 데이터가 존재합니다.")
            if len(saved) == len(self.df_desc) and np.allclose(np.linalg.norm(saved, axis=1, keepdims=True), 1) == True:
                print("데이터 무결성 테스트 통과.")
            else:
                retry = input("데이터 정규화 과정에 오류가 있거나 생성파일이 존재하지 않습니다. 정규화를 다시 시도 하시겠습니까?(y/n) : ")
                if retry == "y":
                    vectors = []

                    for i in range(0, len(self.df_desc), ChatBot.BATCH):
                        batch = self.df_desc[i:i + ChatBot.BATCH]
                        results = ollama.embed(model=ChatBot.MODEL, input=batch)
                        vectors.extend(results['embeddings'])
                        if len(batch) < len(self.df_desc):
                            print(f"임베딩중... {len(vectors)} / {len(self.df_desc)}")

                    desc_array = np.array(vectors, dtype=np.float32)
                    desc_norm = desc_array / np.linalg.norm(desc_array, axis=1, keepdims=True)
                    np.save(ChatBot.DESC_NPY_PATH, desc_norm)
                else:
                    return
        else:
            vectors = []

            for i in range(0, len(self.df_desc), ChatBot.BATCH):
                batch = self.df_desc[i:i + ChatBot.BATCH]
                results = ollama.embed(model=ChatBot.MODEL, input=batch)
                vectors.extend(results['embeddings'])
                if len(batch) < len(self.df_desc):
                    print(f"임베딩중... {len(vectors)} / {len(self.df_desc)}")

            desc_array = np.array(vectors, dtype=np.float32)
            desc_norm = desc_array / np.linalg.norm(desc_array, axis=1, keepdims=True)
            np.save(ChatBot.DESC_NPY_PATH, desc_norm)

    def filter_embed(self, input_norm): # 필터결과와 줄거리 정규화 데이터의 교집합 데이터 추출.
        if not filter_results:
            c.compare_norms(input_norm)
        else:
            saved = np.load(ChatBot.DESC_NPY_PATH)
            filter_norm_intersect = [filter_results[i]['show_id'] for i in range(len(filter_results)) if saved[i] == filter_results[i][f's{i+1}']]

            print(filter_norm_intersect)

    def compare_norms(self, input_norm): # 두 정규화 데이터 비교후 딕셔너리 타입으로 변환후 상위 10개만 추출
        saved = np.load(ChatBot.DESC_NPY_PATH)
        compare_results = saved @ input_norm

        to_dict = {}
        for sid, score in zip(self.ids, compare_results):
            to_dict[sid] = float(score)

        with open(ChatBot.NORM_PATH, mode="w", encoding="utf-8") as f:
            json.dump(to_dict, f, indent=4, ensure_ascii=False)

        top_n = sorted(to_dict, key=to_dict.get, reverse=True)[:10]

        print(top_n)

        return top_n

if __name__ == "__main__":
    c = ChatBot()
    while True:
        user_input = str(input("user : "))
        if user_input in ['q', 'quit', 'exit']:
            print('exit the program.')
            break
        else:
            keyword_result = c.call_gemini(user_input)
            filter_results = c.extract_pd(keyword_result)
            input_norm = c.input_embed(user_input)
            filter_desc = c.filter_embed(filter_results)
            c.save_or_load(filter_desc)
            #top_n = c.compare_norms(input_norm)
            