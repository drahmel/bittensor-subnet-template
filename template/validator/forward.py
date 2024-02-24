# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import bittensor as bt
import random

from template.protocol import Dummy
from template.validator.reward import get_rewards
from template.utils.uids import get_random_uids


async def forward(self):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    # TODO(developer): Define how the validator selects a miner to query, how often, etc.
    # get_random_uids is an example method, but you can replace it with your own.
    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)

    questions = [
        "How did you start your journey as an influencer?",
        "What motivates you to create content?",
        "How do you choose the brands you collaborate with?",
        "What has been your most memorable collaboration so far?",
        "How do you balance your personal life with your influencer responsibilities?",
        "What challenges have you faced in your influencer career?",
        "How do you engage with your audience and build a community?",
        "What tips do you have for someone aspiring to become an influencer?",
        "How do you stay authentic while promoting products or services?",
        "What's your process for creating content?",
        "How do you measure the success of your content?",
        "What trends do you see emerging in your niche?",
        "How do you handle negative feedback or criticism?",
        "What role does social media play in your life as an influencer?",
        "How do you stay updated with the latest trends and tools in your industry?",
        "What's the most important lesson you've learned as an influencer?",
        "How do you manage your time between content creation and other responsibilities?",
        "What's your advice for maintaining mental health in the influencer industry?",
        "How do you ensure that your content resonates with your audience?",
        "What future plans do you have for your influencer career?"
        "What inspires your content creation?",
        "How do you stay true to your personal brand while evolving?",
        "What has been the biggest turning point in your influencer career?",
        "How do you deal with the pressure of constantly being in the public eye?",
        "What's your approach to working with new brands or products?",
        "How do you maintain transparency and trust with your followers?",
        "What role does storytelling play in your content?",
        "How do you navigate the ever-changing algorithms of social media platforms?",
        "What's your strategy for growing your audience?",
        "How do you handle brand partnerships and sponsored content?",
        "What impact do you hope to have on your followers?",
        "How do you differentiate yourself from other influencers in your niche?",
        "What's the most rewarding part of being an influencer?",
        "How do you balance artistic creativity with commercial demands?",
        "What advice do you have for building a loyal and engaged community?",
        "How do you handle the competition within the influencer industry?",
        "What's your process for planning and organizing your content calendar?",
        "How do you keep your content fresh and relevant?",
        "What challenges do you foresee in the future of influencer marketing?",
        "How do you prioritize your mental and physical well-being while managing a demanding schedule?"
    ]

    question = random.choice(questions)

    # The dendrite client queries the network.
    responses = self.dendrite.query(
        # Send the query to selected miner axons in the network.
        axons=[self.metagraph.axons[uid] for uid in miner_uids],
        # Construct a dummy query. This simply contains a single integer.
        synapse=Dummy(dummy_input=self.step, question_input=question),
        # All responses have the deserialize function called on them before returning.
        # You are encouraged to define your own deserialization function.
        deserialize=True,
    )

    # Log the results for monitoring purposes.
    bt.logging.info(f"Received responses: {responses}")

    # TODO(developer): Define how the validator scores responses.
    # Adjust the scores based on responses from miners.
    rewards = get_rewards(self, query=self.step, responses=responses)

    bt.logging.info(f"Scored responses: {rewards}")
    # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
    self.update_scores(rewards, miner_uids)
