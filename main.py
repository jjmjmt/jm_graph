import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import io

# 1. 브라우저 탭 및 앱 제목 설정
st.set_page_config(page_title="영화 유형 나누기", page_icon="🎬", layout="wide")
st.title("🎬 영화 유형 나누기 앱")

# 2. 인터넷 안 터져도 무조건 작동하도록 코드 내부에 데이터 심어두기 (원천 오류 차단)
csv_data = """movieCd,movieNm,openDt,genre,nation,first_scrn,first_show,first_date,peak,first_week_audi,total_audi,days_in_top10
20192481,기생충,2019-05-30,드라마,한국,1948,8253,2019-05-30,1,5359483,10313501,42
20183782,극한직업,2019-01-23,코미디,한국,1971,10231,2019-01-23,1,5261352,16266338,50
20194411,엑시트,2019-07-31,액션,한국,1660,7821,2019-07-31,1,3742105,9426960,38
20188042,독전,2018-05-22,범죄,한국,1133,5214,2018-05-22,1,2135401,5064115,29
20170561,신과함께-죄와 벌,2017-12-20,판타지,한국,1912,9874,2017-12-20,1,5691012,14414658,45
20186281,신과함께-인과 연,2018-08-01,판타지,한국,2235,11024,2018-08-01,1,6791245,12276350,35
20199951,겨울왕국 2,2019-11-21,애니메이션,미국,2648,13241,2019-11-21,1,5981240,13750668,40
20182533,어벤져스: 엔드게임,2019-04-24,액션,미국,2835,14235,2019-04-24,1,7245100,13977602,32
20168754,부산행,2016-07-20,액션,한국,1788,8541,2016-07-20,1,4951200,11567815,28
20150976,베테랑,2015-08-05,액션,한국,1064,5612,2015-08-05,1,3512400,13414484,48
20134803,명량,2014-07-30,액션,한국,1587,7945,2014-07-30,1,6024100,17615919,46
20148569,국제시장,2014-12-17,드라마,한국,966,4852,2014-12-17,1,2112400,14264478,62
20184481,범죄도시2,2022-05-18,범죄,한국,2498,12451,2022-05-18,1,4651200,12693415,36
20224661,범죄도시3,2023-05-31,범죄,한국,2582,13100,2023-05-31,1,5214000,10682813,30
20234675,서울의 봄,2023-11-22,드라마,한국,2328,11542,2023-11-22,1,3124500,13127000,55
20236180,파묘,2024-02-22,미스터리,한국,2367,11840,2024-02-22,1,3689000,11913000,42
"""

try:
    # 문자열 데이터를 데이터프레임으로 신속하게 변환
    df_raw = pd.read_csv(io.StringIO(csv_data.strip()), encoding="utf-8")
    total_count = len(df_raw)
    
    # 3. 데이터 전처리 (결측치 제거 및 첫 주 관객 > 0 조건 적용)
    required_cols = ['first_scrn', 'total_audi', 'days_in_top10', 'first_week_audi']
    df = df_raw.dropna(subset=required_cols).copy()
    df = df[df['first_week_audi'] > 0].copy()
    
    # 상용로그 계산 및 롱런 지수(최대 20) 생성
    df['스크린 수(로그)'] = np.log10(df['first_scrn'])
    df['누적 관객(로그)'] = np.log10(df['total_audi'])
    df['10위권 일수'] = df['days_in_top10']
    df['롱런 지수'] = (df['total_audi'] / df['first_week_audi']).clip(upper=20)
    
    valid_count = len(df)
    st.write(f"📊 **전체 영화 편수:** {total_count}편 | **분석 및 묶음에 사용된 영화 편수:** {valid_count}편")
    
    # 4. 분석 속성 멀티셀렉터 (기본 4개 모두 선택)
    feature_options = ["스크린 수(로그)", "누적 관객(로그)", "10위권 일수", "롱런 지수"]
    st.subheader("⚙️ 분석 설정")
    selected_features = st.multiselect("군집 분석에 사용할 속성을 고르세요 (2개 이상)", options=feature_options, default=feature_options)
    
    if len(selected_features) < 2:
        st.error("⚠️ 속성을 최소 2개 이상 선택해야 분석을 진행할 수 있습니다.")
    else:
        # 5. 표준화 및 K-Means 군집화 (난수 42 고정, 3개 그룹)
        X_scaled = StandardScaler().fit_transform(df[selected_features])
        df['cluster_raw'] = KMeans(n_clusters=3, random_state=42).fit_predict(X_scaled)
        
        # 누적 관객 평균순으로 ㉮, ㉯, ㉰ 매핑
        c_means = df.groupby('cluster_raw')['total_audi'].mean().sort_values(ascending=False)
        rank_mapping = {raw_idx: label for raw_idx, label in zip(c_means.index, ['㉮', '㉯', '㉰'])}
        df['묶음'] = df['cluster_raw'].map(rank_mapping)
        df = df.sort_values(by='묶음')
        
        # 6. Plotly 2차원 산점도
        st.subheader("📈 2차원 산점도")
        col1, col2 = st.columns(2)
        with col1: x_2d = st.selectbox("2차원 X축", options=selected_features, index=0)
        with col2: y_2d = st.selectbox("2차원 Y축", options=selected_features, index=min(1, len(selected_features)-1))
            
        fig_2d = px.scatter(df, x=x_2d, y=y_2d, color='묶음', category_orders={"묶음": ["㉮", "㉯", "㉰"]},
                            hover_name='movieNm', color_discrete_sequence=px.colors.qualitative.Set1)
        st.plotly_chart(fig_2d, use_container_width=True)
        
        # 7. Plotly 3차원 산점도 (속성 3개 미만 시 안내문구)
        st.subheader("🥽 3차원 산점도")
        if len(selected_features) < 3:
            st.info("💡 속성을 3개 이상 선택하면 3차원 산점도가 나타납니다.")
        else:
            col_x, col_y, col_z = st.columns(3)
            with col_x: x_3d = st.selectbox("3차원 X축", options=selected_features, index=0)
            with col_y: y_3d = st.selectbox("3차원 Y축", options=selected_features, index=1)
            with col_z: z_3d = st.selectbox("3차원 Z축", options=selected_features, index=2)
                
            fig_3d = px.scatter_3d(df, x=x_3d, y=y_3d, z=z_3d, color='묶음', category_orders={"묶음": ["㉮", "㉯", "㉰"]},
                               hover_name='movieNm', color_discrete_sequence=px.colors.qualitative.Set1)
            fig_3d.update_traces(marker=dict(size=4))
            st.plotly_chart(fig_3d, use_container_width=True)
            
        # 8. 원래 단위 통계 표
        st.subheader("📋 묶음별 요약 통계")
        stat = df.groupby('묶음').agg(편수=('movieCd','count'), 스크린수=('first_scrn','mean'), 관객수=('total_audi','mean'), 일수=('days_in_top10','mean'), 롱런=('롱런 지수','mean')).reset_index()
        stat.columns = ['묶음', '편수', '스크린 수 평균', '누적 관객 평균', '10위권 일수 평균', '롱런 지수 평균']
        st.dataframe(stat.style.format({'스크린 수 평균':'{:,.1f}', '누적 관객 평균':'{:,.1f}', '10위권 일수 평균':'{:,.1f}', '롱런 지수 평균':'{:,.2f}'}), use_container_width=True)
        
        # 9. 상위 5편 영화 리스트
        st.write("🔝 **묶음별 누적 관객 상위 5편 영화**")
        cols = st.columns(3)
        for idx, label in enumerate(['㉮', '㉯', '㉰']):
            with cols[idx]:
                st.markdown(f"### 묶음 {label}")
                top5 = df[df['묶음'] == label].sort_values(by='total_audi', ascending=False).head(5)
                for r, row in enumerate(top5.itertuples(), 1):
                    st.write(f"{r}. **{row.movieNm}** ({row.total_audi:,}명)")

except Exception as e:
    st.error(f"오류가 발생했습니다: {e}")
