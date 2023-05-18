import rclpy
from rclpy.node import Node

from angel_msgs.msg import InterpretedAudioUserEmotion, SystemTextResponse
import openai
import json
import os
openai.organization = os.getenv("ORG_ID")
openai.api_key = os.getenv("OPENAI_API_KEY")

IN_EMOTION_TOPIC = "user_emotion_topic"
OUT_QA_TOPIC = "system_text_response_topic"

def prompt_gpt(question, model="gpt-3.5-turbo"):
    prompt = """I am wrapping a tourniquet for an injured comrade. Please help. 
Here are the steps for wrapping a tourniquet.
'place tourniquet over affected extremity 2-3 inches above wound site (step 1)
'pull tourniquet tight (step 2)
'apply strap to strap body (step 3)
'turn windless clock wise or counter clockwise until hemorrhage is controlled (step 4)
'lock windless into the windless keeper (step 5)'
'pull remaining strap over the windless keeper (step 6)'
'secure strap and windless keeper with keeper securing device (step 7)'
'mark time on securing device strap with permanent marker (step 8)'
Here are commonly asked questions and answers about tourniquets. Answer only the last question.
Q: How do you find the source of the bleeding?
A: Have the injured person lie down, which will make it easier to locate the exact source of the bleeding.

Q: What should you do after identifying the source of bleeding?
A: Apply direct pressure to the wound. If bleeding does not slow or stop after 15 minutes, consider using a tourniquet.

Q: How should you position the tourniquet?
A: Place the tourniquet on bare skin several inches above the injury, closer to the heart, and avoid placing it directly on a joint. Secure it with a common square knot.

Q: What is a windlass, and how is it used in applying a tourniquet?
A: A windlass is an object used to tighten the tourniquet. Place it on top of the square knot and tie the loose ends of the tourniquet around it with another square knot.

Q: How do you tighten the tourniquet?
A: Twist the windlass until the bleeding stops or is significantly reduced. Secure the windlass by tying one or both ends to the injured person's limb.

Q: How long can a tourniquet be applied for?
A: A tourniquet should not be applied for longer than two hours.

Q: What should you do if the bleeding does not stop after applying a tourniquet?
A: Try twisting the tourniquet more to see if it helps. If not, apply a second tourniquet immediately below the first one without removing the first one.

Q: What are some common mistakes to avoid when applying a tourniquet?
A: Common mistakes include waiting too long to apply a tourniquet, applying it too loosely, not applying a second tourniquet if needed, loosening a tourniquet, and leaving it on for too long.

Q: Who should remove a tourniquet?
A: A tourniquet should only be removed by a healthcare provider in the emergency department.

Q: {}""".format(question)
    payload = {
        "model" : model,
        "messages" : [
            {
                "role": "user",
                "content" : prompt
            }
        ]
    }
    req = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers={"Authorization":"Bearer {}".format(os.getenv("OPENAI_API_KEY"))})
    return json.loads(req.text)['choices'][0]['message']['content'].split("A:")[-1].lstrip()
    

class QuestionAnswerer(Node):
    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.log = self.get_logger()

        parameter_names = [
            IN_EMOTION_TOPIC,
            OUT_QA_TOPIC,
        ]
        set_parameters = self.declare_parameters(
            namespace="",
            parameters=[(p,) for p in parameter_names],
        )
        some_not_set = False
        for p in set_parameters:
            if p.type_ is rclpy.parameter.Parameter.Type.NOT_SET:
                some_not_set = True
                self.log.error(f"Parameter not set: {p.name}")
        if some_not_set:
            raise ValueError("Some parameters are not set.")

        self._in_emotion_topic = \
            self.get_parameter(IN_EMOTION_TOPIC).value
        self._out_qa_topic = \
            self.get_parameter(OUT_QA_TOPIC).value
        self.log.info(f"Input Emotion topic: "
                      f"({type(self._in_emotion_topic).__name__}) "
                      f"{self._in_emotion_topic}")
        self.log.info(f"Output Question-Answer topic: "
                      f"({type(self._out_qa_topic).__name__}) "
                      f"{self._out_qa_topic}")

        # TODO(derekahmed): Add internal queueing to reduce subscriber queue
        # size to 1.
        self.subscription = self.create_subscription(
            InterpretedAudioUserEmotion,
            self._in_emotion_topic,
            self.listener_callback,
            1)

        self._qa_publisher = self.create_publisher(
            SystemTextResponse,
            self._out_qa_topic,
            1
        )

    def get_response(self, user_utterance: str, user_emotion: str):
        '''
        Generate a  response to the utterance, enriched with the addition of
        the user's detected emotion. Inference calls can be added and revised
        here.
        '''
        utterance_words = user_utterance.split()
        # shortened_utterance = \
        #     ' '.join(utterance_words[:4]) + " ... " + \
        #         ' '.join(utterance_words[-4:]) \
        #         if len(utterance_words) >= 8 else user_utterance
        apology_msg = "I'm sorry. I don't know how to answer your statement."
        try: 
            return prompt_gpt(user_utterance)
        except:
            return self._red_font(apology_msg) +\
                f" I understand that you feel \"{self._red_font(user_emotion)}\"."

    def _publish_generated_response(self, utterance: str,
                                    response: str):
        msg = SystemTextResponse()
        msg.utterance_text = utterance
        msg.response = response
        self.log.info(f"Responding to utterance \"{utterance}\" " +\
                      f"with:\n{response}")        
        self._qa_publisher.publish(msg)

    def listener_callback(self, msg):
        '''
        This is the main ROS node listener callback loop that will process
        all messages received via subscribed topics.
        '''  
        utterance = msg.utterance_text
        self.log.info(f"Received utterance:\n\n{utterance}")
        emotion = msg.user_emotion
        response = self.get_response(utterance, emotion)
        self._publish_generated_response(utterance, response)

    def _red_font(self, text):
        return f"\033[91m{text}\033[0m"

def main():
    rclpy.init()
    question_answerer = QuestionAnswerer()
    rclpy.spin(question_answerer)
    question_answerer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
