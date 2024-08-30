import requests
import logging
import json


class TranslateModule:
    def __init__(self, defaultlang=1):
        self.defaultlang = defaultlang

    def translate(self, query, src_lang, dst_lang):
        langs = {1: "eng_Latn", 2: "hin_Deva", 3: "tam_Taml"}
        # url = f"http://127.0.0.1:5000/udaan_project_layout/translate/{langs[src_lang]}/{langs[dst_lang]}"
        url = f"http://10.10.13.2:5000/udaan_project_layout/translate/{langs[src_lang]}/{langs[dst_lang]}"
        # url = f"http://103.42.51.129:5000/udaan_project_layout/translate/{langs[src_lang]}/{langs[dst_lang]}"
        print(query)
        payload = {"sentence": query}
        if dst_lang == 3:
            q2 = query.replace("'", " ")
            query = q2.replace('"', " ")
            q = query.replace("\n", " ")
            payload = {"sentence": q}
            print(q)
        # Headers (add any required headers like API key if necessary)
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
            # "Authorization": "Bearer YOUR_API_KEY"  # Uncomment if API key is needed
        }
        if langs[src_lang] == "eng_Latn" and langs[dst_lang] == "hin_Deva":
            # url = "http://103.42.51.129:5001/udaan_project_layout/translate/en/hi/med,med_comp/0"
            url = "http://10.10.13.2:5001/udaan_project_layout/translate/en/hi/med,med_comp/0"
        # url = "http://127.0.0.1:5001/udaan_project_layout/translate/en/hi/med,med_comp/0"
        try:
            # Make the POST request
            response = requests.post(url, data=payload, headers=headers)
            response.raise_for_status()  # Raise an HTTPError if the HTTP request returned an unsuccessful status code
            translated_data = response.json()  # Parse the JSON response
            print(response.json())
            translated_text = translated_data.get(
                "translation", "Translation key not found"
            )
            print(translated_text)
            # Ensure the key exists in the response
            # if "translated_text" in translated_data:
            #     print("Translated Text:", translated_data["translation"])
            # else:
            #     raise ValueError(
            #         "The expected key 'translation' was not found in the response."
            #     )

        except requests.exceptions.ConnectionError as ce:
            logging.error(f"Connection Error: {ce}")
            raise
        except requests.exceptions.HTTPError as he:
            logging.error(f"HTTP Error: {he}")
            raise
        except requests.exceptions.Timeout as te:
            logging.error(f"Timeout Error: {te}")
            raise
        except requests.exceptions.RequestException as re:
            logging.error(f"Request Exception: {re}")
            raise
        # except ValueError as ve:
        #     logging.error(f"Value Error: {ve}")
        #     raise
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise

        return translated_text


if __name__ == "__main__":
    trans = TranslateModule()
    try:
        trans.translate(
            """**Generated Response:** Based on the provided discharge information, if you have chest pain, you are advised to:
1. Take T.ANGISED 0.6 mg or T.SORBITRATE 5 mg and keep it under your tongue. If no relief, repeat after 5 minutes and report to the nearest doctor for an urgent ECG.
Please note that this advice is specific to your individual case and may not be applicable to everyone. Its essential to consult with your doctor or a medical professional for personalized guidance.
If you have any further questions or concerns, feel free to ask!
**Attribution:** ADVICE @ DISCHARGE :
1. Regular medications & STOP SMOKING.
2. Avoid Alcohol, Heavy exertion and lifting weights.
3. Diet - High fiber, low cholesterol, low sugar (no sugar if diabetic), fruits, vegetables (5 servings
per day).
4. Exercise - Walk at least for 30 minutes daily. Avoid if Chest pain.
5. TARGETS * LDL<70mg/dl *BP - 120/80mmHg * Sugar Fasting - 100mg/dl Post Breakfast – 150mg/dl
* BMI<25kg/m2.
6. IF CHEST PAIN – T.ANGISED 0.6 mg or T.SORBITRATE 5 mg keep under tongue. Repeat if no relief
@ 5 minutes and report to nearest doctor for urgent ECG.
For emergency please contact 0413-2296574 or +91 7867086319""",
            1,
            3,
        )
    except Exception as e:
        print(f"An error occurred during translation: {e}")
