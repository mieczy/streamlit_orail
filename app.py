import streamlit as st

from ml.detection import OrailModel


model = OrailModel()

# layout
INFO_AREA = st.container()
FILE_AREA = st.container()
RESULT_AREA = st.container()
IMAGE_AREA = st.container()

def main():
    with INFO_AREA:
        INFO_AREA.subheader('Model Load')
        INFO_AREA.info('.\Densenet121.pth')

    with FILE_AREA:
        FILE_AREA.subheader('Insert Test Image')
        image_file = FILE_AREA.file_uploader('Select input image',type=['png','jpeg', 'jpg'])
        is_clicked = FILE_AREA.button('inference')
        FILE_AREA.write('---')

    with RESULT_AREA:
        type_col, odd_col = RESULT_AREA.columns(2)
        with type_col:
            type_col.subheader('Result')
        with odd_col:
            odd_col.subheader('Is this OOD Image?')
        
    with IMAGE_AREA:
        IMAGE_AREA.subheader('Simailar Images')
        image1, image2, image3, image4  = IMAGE_AREA.columns(4)

    if is_clicked and image_file is not None:
        result, is_odd = model.detect(image_file)
        type_col.write(result)
        odd_col.write(is_odd)
        odd_col.write('(*OOD score 1-5 and its like confident level)')
        image1.image(image_file, width=200, caption='Input')
        image2.image(image_file, width=200, caption='Similar 1')
        image3.image(image_file, width=200, caption='Similar 2')
        image4.image(image_file, width=200, caption='Similar 3')


if __name__ == '__main__':
    main()
