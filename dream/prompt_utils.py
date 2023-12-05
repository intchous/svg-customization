def get_negtive_prompt_text():
    BASE_PROMPT = ",(((lineart))),((low detail)),(simple),high contrast,sharp,2 bit"
    BASE_NEGPROMPT = "(((text))),(shading),background,noise,dithering,gradient,detailed,out of frame,ugly,error, watermark, stripe"

    neg_prompt = "lowres, bad_anatomy, error_body, error_hair, error_arm, error_hands, bad_hands, error_fingers, bad_fingers, missing_fingers, error_legs, bad_legs, multiple_legs, missing_legs, error_lighting, error_shadow, error_reflection, text, error, extra_digit, fewer_digits, cropped, worst_quality, low_quality, normal_quality, jpeg_artifacts, signature, watermark, username, blurry"

    BASE_NEGPROMPT = BASE_NEGPROMPT + ", " + neg_prompt

    StyleDict = {
        "Illustration": BASE_PROMPT + ",(((vector graphic))),medium detail",

        "Logo": BASE_PROMPT + ",(((centered vector graphic logo))),negative space,stencil,trending on dribbble",

        "Drawing": BASE_PROMPT + ",(((cartoon graphic))),childrens book,lineart,negative space",

        "Artistic": BASE_PROMPT + ",(((artistic monochrome painting))),precise lineart,negative space",

        "Tattoo": BASE_PROMPT + ",(((tattoo template, ink on paper))),uniform lighting,lineart,negative space",

        "Gothic": BASE_PROMPT +
        ",(((gothic ink on paper))),H.P. Lovecraft,Arthur Rackham",

        "Anime": BASE_PROMPT + ",(((clean ink anime illustration))),Studio Ghibli,Makoto Shinkai,Hayao Miyazaki,Audrey Kawasaki",

        "Cartoon": BASE_PROMPT + ",(((clean ink funny comic cartoon illustration)))",

        "Sticker": ",(Die-cut sticker, kawaii sticker,contrasting background, illustration minimalism, vector, pastel colors)",

        "Gold Pendant": ",gold dia de los muertos pendant, intricate 2d vector geometric, cutout shape pendant, blueprint frame lines sharp edges, svg vector style, product studio shoot", "None - prompt only": ""
    }

    return BASE_NEGPROMPT


def get_prompt_text(signature, description_start="a clipart of ", description_end=" "):

    prompt_appd = "(((vector graphic))),(simple),(((white background))),((no outlines)), minimal flat 2d vector icon. lineal color. trending on artstation. professional vector illustration, cartoon, clear, 2d, svg vector style, best quality. trending on dribbble. masterpiece, beautiful detailed, cute, high resolution, intricate detail, hignity 8k wallpaper, detailed"

    prompt = description_start + signature + description_end + prompt_appd

    return prompt
