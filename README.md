# Face mesh application using streamlit and mediapipe
- clone coded from youtube: https://www.youtube.com/watch?v=wyWmWaXapmI
- modified for 
    - bug fix 
    - more functions

## STATUS
 - 60 FPS 영상을 넣었을 때 20FPS 이하로 재생됨. 
 - 물론 파일로 저장하면 원래 속도로 나옴. 그러나, 소리는 없음.
 - Apple의 ARKit에 비해 많이 낮은 품질을 보여 줌.
    - ARKit을 이용한 언리얼의 Live Link Face 앱을 이용하면 정확도 95-99% 정도를 보여준다면, Mediapipe는 30-40% 수준임. (느낌적 느낌)

## TODO
 - Video Face Mesh
    - Video 시작 버튼 추가
    - GPU를 이용하도록 변경
        - 다만, 아직 Windows는 GPU 지원을 안 하는 듯. 
            - https://google.github.io/mediapipe/getting_started/gpu_support.html)
    - 저장파일에 음성 겹쳐넣기


