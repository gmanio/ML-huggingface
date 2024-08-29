# package management

pip list --format=freeze > requirements.txt

# pip list를 사용하면 아래와 같이 requirements.txt 파일에는 패키지 이름과 버전만이 기록이 된다.
# torch==1.13.1+cu116
# torchaudio==0.13.1+cu116
# torchsummary==1.5.1
# torchtext==0.14.1
# torchvision==0.14.1+cu116
# tqdm==4.64.1

pip install -r requirements.txt