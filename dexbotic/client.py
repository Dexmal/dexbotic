from collections import deque
import logging
import requests
import math

import numpy as np
import cv2

logger = logging.getLogger(__name__)


class DexClient:
    def __init__(self,
                 base_url,
                 use_delta=True):
        self.base_url = base_url
        self.use_delta = use_delta

        self.set_init_action()
        self.action_queue = deque()

    def set_init_action(self, action=[0, 0, 0, 0, 0, 0, 0]):
        self.last_act = action

    def act(self,
            observation,
            pormpt):

        if len(self.action_queue) == 0:
            self.acquire_new_action(observation, pormpt)

        action = self.action_queue.popleft()
        self.last_act = action
        return action

    def acquire_new_action(self,
                           observation,
                           prompt):
        images = [observation['image']]

        encoded_images = []
        for image in images:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            ret, encoded_image = cv2.imencode('.png', image)
            encoded_images.append(encoded_image.tobytes())

        ret = requests.post(
            self.base_url + "/process_frame",
            data={"text": prompt},
            files=[("image", _img) for _img in encoded_images],
        )
        response = ret.json().get('response')

        # apply delta if necessary
        last_act = self.last_act
        for action in response:
            if self.use_delta:
                action = self.delta_action(last_act, action)
            else:
                action = np.copy(action)
            self.action_queue.append(action)

            last_act = action

    def delta_action(self, last_action, delta_action):
        original_action = np.copy(last_action)
        original_action[6:] = 0
        action = original_action + delta_action
        action[3:6] = np.where(
            action[3:6] > math.pi,
            action[3:6] - 2 * math.pi,
            action[3:6]
        )
        action[3:6] = np.where(
            action[3:6] < -math.pi,
            action[3:6] + 2 * math.pi,
            action[3:6]
        )
        return action


if __name__ == "__main__":
    # Example usage
    client = DexClient(base_url="http://localhost:7891")
    observation = {
        'image': cv2.imread('test_data/libero_test.png'),
    }
    for i in range(16):
        action = client.act(
            observation,
            "What action should the robot take to put both moka pots on the stove?")
        logger.info("Action taken: %s", action)
