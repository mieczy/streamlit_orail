import streamlit as st
import pandas as pd

from detection import get_model


st.set_page_config(
    page_title="Orail Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)


IMAGE_PATH = 'images/'
LABEL = ['NORMAL', 'AVG', 'MULTITREND', 'HUNTING', 'DRIFT']

res_df = pd.DataFrame(
                {
                    'label': LABEL,
                    'probability': [0,0,0,0,0],
                }
            )


def main():

    init()

    with st.sidebar:
        # model
        st.header('Model Load')
        models = ['Densenet121.pth']
        st.session_state.model = st.selectbox(':green[Selcet model version]', models)

        # image
        st.header('Insert Test Image')
        image_file = st.file_uploader(':green[Select image]',type=['png','jpeg', 'jpg'])
    
        # button
        col1, col2 = st.columns([0.5,1])
        with col1:
            is_clicked_inference = st.button('inference', on_click=inference, args=(image_file,))
        with col2:
            is_clicked_reset = st.button('reset', on_click=reset)

    with st.container():
        result_col, e_col, odd_col = st.columns([0.45, 0.05, 0.5])
        with result_col:
            st.subheader('Result')
            st.dataframe(
                st.session_state.result_df,
                column_config = {
                    'label': 'Label',
                    'probability': st.column_config.ProgressColumn(
                        'Probability',
                        format = '%.2f %%',
                        min_value = 0,
                        max_value = 100,
                    ),
                },
                hide_index = True,
                use_container_width = True,
            )

        with odd_col:
            st.subheader('Is this OOD Image?')
            st.info(f'**{st.session_state.result_ood}**')
            st.write('\* OOD score of 1-2 indicates a confidence level')
        st.empty()
        
    with st.container():
        st.subheader('Similar Images')
        image1, image2, image3, image4 = st.columns(4)
        image1.image(st.session_state.input_image, caption='Input')
        image2.image(st.session_state.similar_image_1, caption='Similar 1')
        image3.image(st.session_state.similar_image_2, caption='Similar 2')
        image4.image(st.session_state.similar_image_3, caption='Similar 3')


def init():
    if 'model' not in st.session_state:
        st.session_state.model = 'Densenet121.pth'
   
    if 'result_df' not in st.session_state:
        st.session_state.result_df = res_df

    if 'result_ood' not in st.session_state:
        st.session_state.result_ood = ' '
    
    default_img = IMAGE_PATH + 'default.png'
    if 'input_image' not in st.session_state:
        st.session_state.input_image = default_img
   
    if 'similar_image_1' not in st.session_state:
        st.session_state.similar_image_1 = default_img
    
    if 'similar_image_2' not in st.session_state:
        st.session_state.similar_image_2 = default_img

    if 'similar_image_3' not in st.session_state:
        st.session_state.similar_image_3 = default_img
   

def inference(image_file):
    if image_file is not None:
        try: 
            model = get_model(st.session_state.model)
            t, label_probabilities, is_odd = model.detect(image_file)
            st.session_state.result_df['probability'] = label_probabilities  
            st.session_state.result_ood = is_odd
            st.session_state.input_image = image_file
            if t == 2:
                st.session_state.similar_image_1 = image_file
                st.session_state.similar_image_2 = image_file
                st.session_state.similar_image_3 = image_file
            else:
                st.session_state.similar_image_1 = IMAGE_PATH + LABEL[t].lower() + '/1.png'
                st.session_state.similar_image_2 = IMAGE_PATH + LABEL[t].lower() + '/2.png'
                st.session_state.similar_image_3 = IMAGE_PATH + LABEL[t].lower() + '/3.png'

        except Exception as e:
            reset()
            st.error('Sorry, there was an issue while processing the request. Please refresh the page to reload the app.')


def reset():
        for key in st.session_state.keys():
            del st.session_state[key]
        init()
    

if __name__ == '__main__':
    main()
