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
    listed_in: list[str] = Field(
        description="장르. 없다면 빈리스트"
    )

class ChatBot:

    NETFLIX_CSV_PATH = "/Users/d.h/dev/ai_chatbot/data_file/netflix_titles.csv"
    DESC_NPY_PATH = "/Users/d.h/dev/ai_chatbot/data_file/netflix_desctiption.npy"
    PD_PATH = "/Users/d.h/dev/ai_chatbot/data_file/netflix_pd.json"
    VEC_PATH = "/Users/d.h/dev/ai_chatbot/data_file/netflix_vec.json"
    API_KEY = os.environ["GEMINI_API_KEY"]
    MODEL = "bge-m3"
    BATCH = 100

    def __init__(self): # 생성자 생성
        pass
    
    def dataframe(self): # 넷플릭스 csv파일 데이터프레임화, 특정 설명, show_id부분 리스트로 변환후 반환.
        df = pd.read_csv(ChatBot.NETFLIX_CSV_PATH)
        df = df.fillna("Unknown")
        descript = df["description"].tolist()
        ids = df["show_id"].tolist()
        
        return df, descript, ids
    
    def call_gemini(self, user_input): # gemini api 호출해서 사용자 입력 카테고리에 맞게 분류.
        genres = sorted(
            df['listed_in'].str.split(',').explode().str.strip().unique()
        )

        prompt = (
            f"""
            사용자가 찾고싶은 영화나 TV프로그램을 자연어 방식으로 입력하면
            너는 {user_input}을 각 카테고리에 맞는 키워드를 넣어주는 역활이야.
            장르부분은 {genres}안에 있는 문자열으로만 넣어줘.
            """
        )

        client = genai.Client(api_key=ChatBot.API_KEY)

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
            print("60초 대기후 재시도...")
            time.sleep(61)
            for i in range(3):
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
                    if keyword_result:
                        return keyword_result
                except Exception as e:
                    print(f"API Call Failure: {e}")
                    print("60초 대기후 재시도...")
                    time.sleep(61)
                    continue
                else:
                    print("api 요청 성공.")

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
        if keyword_result['listed_in']:
            l = "|".join(keyword_result['listed_in'])
            df_c = df_c[df_c['listed_in'].str.contains(l, case=False, na=False)]

        results = df_c.to_dict(orient="records")
            
        with open(ChatBot.PD_PATH, mode="w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
    def input_embed(self, user_input): # 사용자 입력 정규화
        result = ollama.embed(model=ChatBot.MODEL, input=user_input)
        input_array = np.array(result["embeddings"][0], dtype=np.float32)
        input_length = input_array / np.linalg.norm(input_array)

        if np.allclose(np.linalg.norm(input_length), 1) == True:
            print(f"사용자 입력 데이터 정규화 완료: {input_length.shape}")
            return input_length
        else:
            retry = input("사용자 입력 데이터 정규화 오류. 다시 시도 하시겠습니까?(y/n) : ")
            if retry == "y":
                re = c.input_embed(user_input)
                return re
            else:
                return

    def desc_embed(self, descript): # 설명부분 정규화.
        vectors = []
        
        for i in range(0, len(descript), ChatBot.BATCH):
            batch = descript[i:i + ChatBot.BATCH]
            result = ollama.embed(model=ChatBot.MODEL, input=batch)
            vectors.extend(result["embeddings"])
            if len(batch) < len(descript):
                print(f"임베딩중... {len(vectors)} / {len(descript)}")

        descript_array = np.array(vectors, dtype=np.float32)
        descript_length = descript_array / np.linalg.norm(descript_array, axis=1, keepdims=True)

        np.save(descript_length, ChatBot.DESC_NPY_PATH)

        return

    def save_or_load(self, descript): # 설명 부분 정규화 데이터 존재 유무 확인 및 데이터 무결성 확인.
        if os.path.exists(ChatBot.DESC_NPY_PATH):
            saved = np.load(ChatBot.DESC_NPY_PATH)
            print("저장된 정규화 데이터가 존재합니다.")
            if len(saved) == len(descript) and np.allclose(np.linalg.norm(saved, axis=1, keepdims=True), 1) == True:
                print("데이터 무결성 테스트 통과.")
            else:
                retry = input("데이터 정규화 오류. 정규화를 다시 시도 하시겠습니까?(y/n) : ")
                if retry == "y":
                    re = c.desc_embed(descript)
                    return re
                else:
                    return
        else:
            make = c.desc_embed(descript)
            return make

    def compare_vectors(self, input_length, ids):
        saved = np.load(ChatBot.DESC_NPY_PATH)
        compare_results = saved @ input_length

        to_dict = {}
        for sid, score in zip(ids, compare_results):
            to_dict[sid] = float(score)

        with open(ChatBot.VEC_PATH, mode="w", encoding="utf-8") as f:
            json.dump(to_dict, f, indent=4, ensure_ascii=False)

        top_n = sorted(to_dict, key=to_dict.get, reverse=True)[:10]
        return top_n

    def vec_compare_pd(self, top_n):
        results = df[df['show_id'].isin(top_n)]

        for i in results:
            print(f"{i}\n")
        
        

# 사용자 입력, 영화 설명부분 ollama로 임베딩후 행 길이 맞추기

# 두 벡터 비교후 상위 N개 추출

# show_id로 두 추출 데이터 교집합 데이터 추출

# gemini api로 ai스럽게 답변

if __name__ == "__main__":
    c = ChatBot()
    df, descript, ids = c.dataframe()
    while True: # 사용자 입력
        user_input = str(input("user : "))
        if user_input in ["q", "quit", "exit"]:
            print("quit the program...")
            break
        else:
            keyword_result = c.call_gemini(user_input)
            c.extract_pd(keyword_result)
            input_length = c.input_embed(user_input)
            c.save_or_load(descript)
            top_n = c.compare_vectors(input_length, ids)
            c.vec_compare_pd(top_n)
