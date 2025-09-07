import numpy as np
import matplotlib.pyplot as plt
from skimage import color, measure
from skimage.filters import gaussian, threshold_otsu
from skimage.transform import resize
from tensorflow.keras.models import load_model
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class ECG:
    def __init__(self, model_path, label_encoder_path, input_length=255):
        # Load trained CNN model
        self.model = load_model(model_path)
        # Load label encoder
        self.label_encoder = joblib.load(label_encoder_path)
        # CNN expected input length (255 features)
        self.input_length = input_length

    def getImage(self, image):
        from PIL import Image
        import numpy as np
        img = np.array(Image.open(image))
        return img

    def GrayImgae(self, image):
        image_gray = color.rgb2gray(image)
        image_gray = resize(image_gray, (1572, 2213))
        return image_gray

    def DividingLeads(self, image):
        # Divide image into 12 standard leads only
        Lead_1 = image[300:600, 150:643]
        Lead_2 = image[300:600, 646:1135]
        Lead_3 = image[300:600, 1140:1625]
        Lead_4 = image[300:600, 1630:2125]
        Lead_5 = image[600:900, 150:643]
        Lead_6 = image[600:900, 646:1135]
        Lead_7 = image[600:900, 1140:1625]
        Lead_8 = image[600:900, 1630:2125]
        Lead_9 = image[900:1200, 150:643]
        Lead_10 = image[900:1200, 646:1135]
        Lead_11 = image[900:1200, 1140:1625]
        Lead_12 = image[900:1200, 1630:2125]

        Leads = [Lead_1, Lead_2, Lead_3, Lead_4, Lead_5, Lead_6,
                 Lead_7, Lead_8, Lead_9, Lead_10, Lead_11, Lead_12]

        # Plot and save images
        fig, ax = plt.subplots(4, 3, figsize=(10, 10))
        x_counter = y_counter = 0
        for i, lead in enumerate(Leads):
            ax[x_counter][y_counter].imshow(lead)
            ax[x_counter][y_counter].axis('off')
            ax[x_counter][y_counter].set_title(f"Lead {i+1}")
            y_counter += 1
            if (i+1) % 3 == 0:
                x_counter += 1
                y_counter = 0
        fig.savefig('Leads_1-12_figure.png')
        return Leads

    def PreprocessingLeads(self, Leads):
        fig, ax = plt.subplots(4, 3, figsize=(10, 10))
        x_counter = y_counter = 0
        for i, lead in enumerate(Leads):
            gray = color.rgb2gray(lead)
            blurred = gaussian(gray, sigma=1)
            thresh = threshold_otsu(blurred)
            binary = blurred < thresh
            binary = resize(binary, (300, 450))
            ax[x_counter][y_counter].imshow(binary, cmap='gray')
            ax[x_counter][y_counter].axis('off')
            ax[x_counter][y_counter].set_title(f"Preprocessed Lead {i+1}")
            y_counter += 1
            if (i+1) % 3 == 0:
                x_counter += 1
                y_counter = 0
        fig.savefig('Preprocessed_Leads_1-12_figure.png')

    def SignalExtraction_Scaling(self, Leads):
        all_scaled = []
        for lead in Leads:
            gray = color.rgb2gray(lead)
            blurred = gaussian(gray, sigma=0.7)
            thresh = threshold_otsu(blurred)
            binary = blurred < thresh
            contours = measure.find_contours(binary, 0.8)
            if len(contours) == 0:
                contour = np.zeros((255, 2))
            else:
                contour = sorted(contours, key=lambda c: c.shape[0], reverse=True)[0]
                contour = resize(contour, (255, 2))
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(contour)[:, 0]
            all_scaled.extend(scaled)

        # Take first 255 features only for prediction
        df_scaled = pd.DataFrame([all_scaled[:255]])
        df_scaled.to_csv('Final_1DFeatures.csv', index=False)
        return df_scaled

    def CombineConvert1Dsignal(self):
        df_final = pd.read_csv('Final_1DFeatures.csv')
        return df_final

    def ModelLoad_predict(self, final_df):
        X = final_df.values.flatten()
        X_resized = np.resize(X, (1, self.input_length, 1))
        pred_probs = self.model.predict(X_resized)
        class_index = np.argmax(pred_probs)
        condition = self.label_encoder.inverse_transform([class_index])[0]
        return condition

    def predict(self, image_path):
        img = self.getImage(image_path)
        leads = self.DividingLeads(img)
        self.PreprocessingLeads(leads)
        df_features = self.SignalExtraction_Scaling(leads)
        df_final = self.CombineConvert1Dsignal()
        prediction = self.ModelLoad_predict(df_final)
        return prediction
