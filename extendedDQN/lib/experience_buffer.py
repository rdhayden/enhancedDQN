import collections
import numpy as np
from xxhash import xxh64
import pickle
import zlib

Experience = collections.namedtuple(
    "Experience", field_names=["state", "action", "reward", "done", "new_state"]
)

ExperienceCompressed = collections.namedtuple(
    "Experience", field_names=["state_key", "action", "reward", "done", "new_state_key"]
)


class ExperienceBuffer:
    def __init__(self, capacity, n_steps, gamma):
        assert n_steps > 0
        self.buffer = collections.deque(maxlen=capacity)
        self.temp_buffer = collections.deque(maxlen=n_steps)
        self.gamma = gamma
        self.n_steps = n_steps
        self._imgs = {}
        self._capacity = capacity

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.temp_buffer.append(experience)
        if experience.done:
            self._make_n_step_experience()
            self.temp_buffer.clear()
        elif len(self.temp_buffer) == self.n_steps:
            self._make_n_step_experience()

    def _make_n_step_experience(self):
        Rt = 0 # initialise the total return
        state_key = self.put_image(self.temp_buffer[0].state)
        new_state_key = self.put_image(self.temp_buffer[-1].new_state)
        for i in range(len(self.temp_buffer)):
            Rt += (self.gamma ** i) * self.temp_buffer[i].reward

        if len(self.buffer) == self._capacity:
            # remove experience from the back of the queue, this would be done by default when adding the next experience
            # however we still need to deal with the states associated with the experience so ...
            # check if the states associated with the experience at the back of the queue are used by other experiences,
            # if not delete the states to reclaim memory, otherwise -1 from the count as this experience no longer
            # references them. This is effectively a garbage collector for unused states in the _imgs dict
            compressed_experience = self.buffer.popleft()
            if self._imgs[compressed_experience.state_key][1] == 1:
                del self._imgs[compressed_experience.state_key]
            else:
                self._imgs[compressed_experience.state_key][1] -= 1
            if self._imgs[compressed_experience.new_state_key][1] == 1:
                del self._imgs[compressed_experience.new_state_key]
            else:
                self._imgs[compressed_experience.new_state_key][1] -= 1

        self.buffer.append(
            ExperienceCompressed(
                state_key,
                self.temp_buffer[0].action,
                Rt,
                self.temp_buffer[-1].done,
                new_state_key,
            )
        )

    def put_image(self, img):
        # compress the image (which is a state) to reduce memory requirements. Also store the images seperately in a dictionary
        # so that we don't have to store a state in one experience as an end state and again in the next experience as a start state
        # which would use a lot of extra memory. It's also possible for the same states to be visited from multiple previous states. 
        # This ensures we only actually store a given video frame once and compress it to minimise the buffer memory requirements
        compressed_img = zlib.compress(pickle.dumps(img))
        key = xxh64(compressed_img).hexdigest()
        if key in self._imgs:
            self._imgs[key][1] += 1
        else:
            self._imgs[key] = [compressed_img, 1]
        return key

    def sample(self, batch_size):
        # sample experiences uniformly from the buffer
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        state_keys, actions, rewards, dones, next_state_keys = zip(
            *[self.buffer[idx] for idx in indices]
        )

        # lookup and decompress the images associated with each state
        states = [
            pickle.loads(zlib.decompress(self._imgs[key][0])) for key in state_keys
        ]
        next_states = [
            pickle.loads(zlib.decompress(self._imgs[key][0])) for key in next_state_keys
        ]

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.uint8),
            np.array(next_states),
        )
