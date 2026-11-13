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

    BASE_DIR = Path(__file__).resolve().parent.parent
    OUTPUT_DIR = BASE_DIR / "output"
    OUTPUT_DIR.mkdir(exist_ok=True)

    #NETFLIX_CSV_PATH = "/Users/d.h/dev/ai_chatbot/data_file/netflix_titles.csv"
    #DESC_NPY_PATH = "/Users/d.h/dev/ai_chatbot/data_file/netflix_description.npy"
    #PD_PATH = "/Users/d.h/dev/ai_chatbot/data_file/netflix_pd.json"
    #VEC_PATH = "/Users/d.h/dev/ai_chatbot/data_file/netflix_vec.json"

    NETFLIX_CSV_PATH = BASE_DIR / "data_file" / "netflix_titles.csv"
    DESC_NPY_PATH = OUTPUT_DIR / "netflix_description.npy"
    PD_PATH = OUTPUT_DIR / "netflix_pd.json"
    VEC_PATH = OUTPUT_DIR / "netflix_vec.json"

    API_KEY = os.environ["GEMINI_API_KEY"]
    MODEL = "bge-m3"
    BATCH = 100

    def __init__(self):
        pass
    
    def dataframe(self): # 넷플릭스 csv파일 데이터프레임화, 줄거리, show_id부분 리스트로 변환.
        df = pd.read_csv(ChatBot.NETFLIX_CSV_PATH)
        df = df.fillna("Unknown")
        descript = df["description"].tolist()
        ids = df["show_id"].tolist()
        
        return df, descript, ids
    
    def call_gemini(self, user_input): # gemini api 호출해서 사용자 입력을 카테고리에 맞게 분류.
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
                model='gemini-3.5-flash',
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
                        model='gemini-3.5-flash',
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

    def desc_embed(self, descript): # 줄거리 부분 정규화.
        vectors = []
        
        for i in range(0, len(descript), ChatBot.BATCH):
            batch = descript[i:i + ChatBot.BATCH]
            result = ollama.embed(model=ChatBot.MODEL, input=batch)
            vectors.extend(result["embeddings"])
            if len(batch) < len(descript):
                print(f"임베딩중... {len(vectors)} / {len(descript)}")

        descript_array = np.array(vectors, dtype=np.float32)
        descript_length = descript_array / np.linalg.norm(descript_array, axis=1, keepdims=True)

        np.save(ChatBot.DESC_NPY_PATH, descript_length)

        return

    def save_or_load(self, descript): # 줄거리 부분 정규화 데이터 존재 유무 및 데이터 무결성 확인.
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

    def compare_vectors(self, input_length, ids): # 두 정규화 데이터 비교후 딕셔너리 타입으로 변환후 상위 10개만 추출
        saved = np.load(ChatBot.DESC_NPY_PATH)
        compare_results = saved @ input_length

        to_dict = {}
        for sid, score in zip(ids, compare_results):
            to_dict[sid] = float(score)

        with open(ChatBot.VEC_PATH, mode="w", encoding="utf-8") as f:
            json.dump(to_dict, f, indent=4, ensure_ascii=False)

        top_n = sorted(to_dict, key=to_dict.get, reverse=True)[:10]

        return top_n

    def vec_compare_pd(self, top_n, df): # pandas로 뽑은 데이터와 정규화 비교 데이터의 교집합 확인. 교집합 없을 시 정규화 상위 10개를 답변으로 선택.
        #results = df[df['show_id'].isin(top_n)]

        with open(ChatBot.PD_PATH, mode="r") as f:
            results_pd = json.load(f)

        #top_dict = {}
        #for sid, score in zip():
        results_1 = []
        results_2 = []

        if not results_pd:
            results_1 = [df.iloc[i] for i in range(len(df['show_id'])) if df.loc[i, 'show_id'] in top_n]
        else:
            results_2 = [results_pd[i] for i in range(len(results_pd)) if results_pd[i]['show_id'] in top_n]

        # 1. top10과 pd_data가 서로 겹치는게 없을때(정확한 데이터가 입력에 없으면 top10이 우선순위)
        # 2. 사용자가 입력에 정확한 데이터를 명시했는데 top10은 줄거리만 비교해서 뽑은거기 때문에 pd_data가 우선순위?
        # 3. gemini api 규칙 좀더 명시화 하기(특수문자 안보이게, ex좀더 자연스럽게 다듬기)
        # 4. db에 show_id, description과 netflix.csv psycopg로 넣어주고 코드와 연결해서 데이터 다루기.

        #print(type(results))
        #print(results)
        return results_1, results_2

    def answer_with_gemini(self, results_1, results_2): # gemini api로 분류한 넷플릭스 데이터를 ai chatbot 처럼 답변출력.
        ex = {
            "영화 제목": "영화제목",
            "감독": "감독이름",
            "출연진": "출연진 이름",
            "국가": "만든 국가 이름",
            "출시연도": "만든 연도",
            "평점": "평점",
            "상영시간": "상영길이",
            "장르": "장르",
            "줄거리": "줄거리 짧게"
        }

        prompt = (
            f"사용자가 원하는 영화를 찾고싶어서 입력을 한 결과물이 {results_1}에 저장되어있어.",
            f"이걸 {ex}을 참고하는데 딕셔너리 형식 그대로 출력하지 말고 넷플릭스 ai 챗봇처럼 깔끔하게 출력을 해줘.",
            f"만약 {results_1}과 {results_2} 모두가 비어있다면 찾는 정보가 없다는 식으로 말을 해서 다른 입력을 하게 도와줘"
        )

        client = genai.Client(api_key=ChatBot.API_KEY)

        try:
            response = client.models.generate_content(
                model='gemini-3.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0,
                ),
            )
            print(response.text)
            return
        
        except Exception as e:
            print(f"API Call Failure: {e}")
            print("60초 대기후 재시도...")
            time.sleep(61)
            for i in range(3):
                try:
                    response = client.models.generate_content(
                        model='gemini-3.5-flash',
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0,
                        ),
                    )
                    print(response.text)
                    return
                except Exception as e:
                    print(f"API Call Failure: {e}")
                    print("60초 대기후 재시도...")
                    time.sleep(61)
                    continue
                else:
                    print("api 요청 성공.")


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
            results_1, results_2 = c.vec_compare_pd(top_n, df)
            c.answer_with_gemini(results_1, results_2)
