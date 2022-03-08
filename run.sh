wget https://github.com/phuongnm-bkhn/legal_text_retrieval/raw/master/data/zac2021-ltr-data.zip
unzip ./zac2021-ltr-data.zip
rm ./zac2021-ltr-data.zip

mv ./zac2021-ltr-data ./dataset

pip install -r requirements.txt

streamlit run main.py
