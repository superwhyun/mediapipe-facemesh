from numpy.lib.function_base import _median_dispatcher
import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile 
import time
from PIL import Image

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

DEMO_IMAGE = 'demo.png'
DEMO_VIDEO = 'demo.mp4'

st.title('Face Mesh Application using Mediapipe')

# 사이드바 생성
# - 멀티라인 text는 """ 블라블라 """를 이용. 얘는 주석이 아님
# - 사이드바의 스타일을 이렇게 만드네... 이건 좀 더 공부해 봐야겠음.
# unsafe_allow_html 옵션--> 기본적으로 markdown에서 HTML 문법 사용을 허락하지 않는데, 이 값을 True로 하면 html tag를 사용가능함.
#   - 사용못하게 하는 이유는 
#       we refer to HTML as “unsafe” as a keyword argument, 
#       to highlight the fact that you can run JavaScript within the widget.
#       We’re highlighting the fact that Streamlit itself will not be able to prove the code’s safety, 
#       since a malicious 3rd-party could inject code into your website, 
#       or the Streamlit app creator might write code in such a way as to allow for code injection attacks 11. 
#       So by default, we disable evaluating code inside of an HTML snippet, 
#       but allow a user to set the keyword argument to say “I understand that this is potentially risky, 
#       but I want to do it anyway”
#       - https://discuss.streamlit.io/t/why-is-using-html-unsafe/4863
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width:350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width:350px
        margin-left: -350px
    }
    </style>
    """, 
    unsafe_allow_html=True,
)
st.sidebar.title('FaceMesh Sidebar')
st.sidebar.subheader('parameters')


# 이건 뭐하는건가?
@st.cache()

# inter는 interpolation
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim=None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = width/float(w)
        dim = (int (w*r), height)
    
    else:
        r = width/float(w)
        dim = (width, int(h*r))
    
    #resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


app_mode = st.sidebar.selectbox('Choose the App mode', 
    ['About', 'Run on Image', 'Run on Video'], index=2
)



if app_mode == 'About':

    st.markdown(
        """
        About this application \n
        We gonna use **MediaPipe**. 
        I've just followed the youtube content
        It's awesome.
        Especially, the first guy have very funny english pronounciation.
        """, 
        unsafe_allow_html=True,
    )

    st.video('https://youtu.be/wyWmWaXapmI')

elif app_mode == "Run on Image":
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width:350px
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width:350px
            margin-left: -350px
        }
        </style>
        """, 
        unsafe_allow_html=True,
    )

    st.markdown("**Detected faces**")
    kpi1_text = st.markdown("0")

    max_faces = st.sidebar.number_input('Max num of Face', value=2, min_value=1)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min detection confidence', min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown('---')

    img_file_buffer = st.sidebar.file_uploader("Upload your image", type=["jpg","png", "jpeg"])
    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))
    else:
        demo_image = DEMO_IMAGE
        image = np.array(Image.open(demo_image))

    st.sidebar.text('Original Image')
    st.sidebar.image(image)

    face_count = 0

    ## Dashboard
    with mp_face_mesh.FaceMesh(
        static_image_mode = True,
        max_num_faces=max_faces,
        min_detection_confidence=detection_confidence) as face_mesh:
            results=face_mesh.process(image)
            out_image=image.copy()

            if results is not None and results.multi_face_landmarks is not None:
                ## Landmark Drawing
                for face_landmarks in results.multi_face_landmarks:
                    face_count+=1
                    mp_drawing.draw_landmarks(
                        image = out_image,
                        landmark_list = face_landmarks,
                        connections = mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec = drawing_spec)
                    kpi1_text.write(f"<h1 style='text-align:center; color:red;'> {face_count}</h1>", unsafe_allow_html=True)
                st.subheader('Output Image')
                if out_image is not None: 
                    st.image(out_image, use_column_width=True)

elif app_mode == "Run on Video":    

    # st.set_option('deprecation.showfileUploaderEncoding', False)
    use_webcam = st.sidebar.button('Use webcam')
    record = st.sidebar.checkbox("Record Video", value=True)

    if record:
        st.checkbox("Recording")

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width:350px
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width:350px
            margin-left: -350px
        }
        </style>
        """, 
        unsafe_allow_html=True,
    )

    st.markdown("**Detected faces**")
    kpi1_text = st.markdown("0")

    max_faces = st.sidebar.number_input('Max num of Face', value=2, min_value=1)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min detection confidence', min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown('---')    
    tracking_confidence = st.sidebar.slider('Min tracking confidence', min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown('---')

    st.markdown("## Output ")

    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", "avi", "asf"])
    tffile = tempfile.NamedTemporaryFile(delete=False)


    ## Input video
    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tffile.name = DEMO_VIDEO
    else:
        tffile.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tffile.name)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))

    ## Recording
    # codec = cv2.VideoWriter_fourcc('V','P','0','9') # => VP9 코덱 
    # codec = cv2.VideoWriter_fourcc('M','J','P','G') # => MJPEG 코덱
    # Windows에서는 이렇게 하는게 가장 안정적이라고 하는 글을 봐서 그렇게 함. --> 동작함.
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', codec, fps_input, (width, height))


    st.sidebar.text('Input Video')
    st.sidebar.video(tffile.name)

    

    fps =0
    i=0

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.markdown("**Frame rate**")
        kpi1_text = st.markdown("0")

    with kpi2:
        st.markdown("**Detected Faces**")
        kpi2_text = st.markdown("0")

    with kpi3:
        st.markdown("**Image width**")
        kpi3_text = st.markdown("0")


    st.markdown("<hr/>", unsafe_allow_html=True)

    ## Dashboard
   
    while vid.isOpened():
        prevTime=0
        with mp_face_mesh.FaceMesh(
                
                max_num_faces=max_faces,
                min_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence) as face_mesh:
            i +=1
            ret, frame = vid.read()
            if not ret:
                break
                        
            results=face_mesh.process(frame)
            frame.flags.writeable = True
            

            face_count = 0
            if results.multi_face_landmarks:
                ## Landmark Drawing
                for face_landmarks in results.multi_face_landmarks:
                    face_count+=1
                    mp_drawing.draw_landmarks(
                        image = frame,
                        landmark_list = face_landmarks,
                        connections = mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec = drawing_spec,
                        connection_drawing_spec=drawing_spec)
                
            curTime = time.time()
            fps = 1/(curTime-prevTime)
            prevTime = curTime

            if record:
                # print(f"writing frame: {i}")
                out.write(frame)

            kpi1_text.write(f"<h1 style='text-align:center; color=red;'>{int(fps)}</h1>", unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style='text-align:center; color=red;'>{face_count}</h1>", unsafe_allow_html=True)
            kpi3_text.write(f"<h1 style='text-align:center; color=red;'>{width}</h1>", unsafe_allow_html=True)

            frame=cv2.resize(frame, (0,0), fx=0.8, fy=0.8)
            frame=image_resize(image=frame, width=640)
            stframe.image(frame, channels='BGR', use_column_width = True)
    
    # print('Play DONE')
    vid.release()
    out.release()

    output_video = open('output.avi', "rb")
    out_bytes = output_video.read()
    st.video(out_bytes)












