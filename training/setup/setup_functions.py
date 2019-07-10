#
# Copyright 2018-2019 IBM Corp. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import requests
import sys
import json


class InstanceHandler:

    def __init__(self, iam_access_token):
        assert iam_access_token is not None, \
            'Parameter access token cannot be None'
        self.iam_access_token = iam_access_token
        with open('setup/config.json') as config_file:
            self.data = json.load(config_file)

    def available_instance(self, which_resource, resource_id): # noqa
        """
        This function retrieves available instances of
        specified service and prompts user to select the instance
        to be used for training.
        1. Available service resource plan ids of instances are
           checked against the standard service resource plan id
           values corresponding to the provided service and resource id.
        2. Displays the available resource plan ids of instances
           under the chosen service.
        3. Prompts user to enter the index of the displayed instances
           of their choice. User can also choose to create a new instance.
        4. If there is no available instance, user will be taken
           directly to instance creation step.
        :param which_resource: under which service, standard service
               resource plan id needs to be searched.
        :param resource_id: under which resource, availability of
               instances need to be searched.
        :return: List of existing instances, guids and
                 selected option. Exit on error
        """
        headers = {
            'Authorization': self.iam_access_token,
        }
        response = requests.get("https://resource-controller."
                                "cloud.ibm.com/v2/resource_instances",
                                headers=headers)
        if response.status_code == 200:
            response = response.json()
            # list to append existing instances
            existing_instances = []
            # list to append existing guids
            existing_guids = []
            count = 0
            # retrieve instances available under
            for index, value in enumerate(response['resources'], start=1):
                # check instances under the specified resource id
                if value['resource_group_id'] == resource_id:
                    for index_1, plan in \
                            enumerate(value['plan_history'], start=1):
                        for key, values in \
                                enumerate(self.data[which_resource].items()):
                            if plan['resource_plan_id'] == values[1]:
                                count += 1
                                print("{:2d}. Instance Name: {}   |  "
                                      "Instance Location: {}  | "
                                      "Instance Plan: {} ".
                                      format(int(count), value['name'],
                                             value['region_id'], values[0]))
                                existing_instances.append(value['name'])
                                existing_guids.append(value['guid'])
            # Adding create new instance option.
            if len(existing_instances) > 0:
                print('{:2d}. {}'.format(int(count) + 1,
                                         '* Create New Instance *'))
                existing_instances.append('Create New Instance')
                existing_guids.append('Create New Guid')
            else:
                # Default instance create option when no instances are found
                print('{:2d}. {}'.format(1, 'Create New Instance'))
            # Prompt user to input choice and return lists and choice.
            if len(existing_instances) > 0 and \
                    existing_instances[0] != 'Create New Instance':
                while True:
                    instance_option = input("[PROMPT] Your selection:  ") \
                                      .strip()
                    if not instance_option.isdigit() or \
                       int(instance_option) < 1 or \
                       (int(instance_option) >= (len(existing_instances) + 1)):
                        print("[MESSAGE] Enter a number between 1 and "
                              "{}.".format(len(existing_instances)))
                        continue
                    else:
                        return existing_instances, instance_option, \
                               existing_guids
            else:
                existing_instances = ['Create New Instance']
                instance_option = '1'
                existing_guids = ['Create New Guid']
                return existing_instances, instance_option, existing_guids

        else:
            print("[DEBUG] Failing with status code:", response.status_code)
            print("[DEBUG] Reason for failure:", response.reason)
            sys.exit()

    def wml_key_check(self, instance_guid): # noqa
        """
        This function:
        1. Retrieves list of keys under provided instance guid.
        2. Prompts user choice to choose from existing
           keys or create new key
        :param instance_guid: guid of instance under
               which presence of keys need to be searched.
        :return: list of existing keys, guid of keys and user
               choice. Exit when error occurs in key retrieval request.
        """
        headers = {
            'Authorization': self.iam_access_token,
        }
        available_keys = requests.get("https://resource-controller.cloud."
                                      "ibm.com/v2/resource_keys",
                                      headers=headers)
        if available_keys.status_code == 200:
            available_keys = available_keys.json()
            # list for storing existing keys and list
            existing_keys = []
            existing_keys_guid = []
            count = 0
            for index, key in enumerate(available_keys['resources'], start=1):
                try:
                    # Check if provided instance guid matches with current
                    if key['credentials']['instance_id'] == instance_guid:
                        count += 1
                        print("{:2d}. {}".format(count, key['name']))
                        existing_keys.append(key['name'])
                        existing_keys_guid.append(key['guid'])
                except KeyError:
                    continue
            print('{:2d}. {}'.format(count + 1,
                                     '* Create New Service Credentials *'))
            existing_keys.append('Create New Key')

            if len(existing_keys) > 0:
                while True:
                    key_option = input("[PROMPT] Your selection:  ").strip()
                    if not key_option.isdigit() or \
                       int(key_option) < 1 or \
                       int(key_option) >= (len(existing_keys) + 1):
                        print("[MESSAGE] Enter a number between 1 and "
                              "{}.".format(len(existing_keys)))
                        continue
                    else:
                        return existing_keys, key_option, existing_keys_guid
            else:
                existing_keys = ['Create New Key']
                key_option = '1'
                existing_keys_guid = []
                return existing_keys, key_option, existing_keys_guid
        else:
            print('[DEBUG] key not present')
            print("[DEBUG] Failing with status code:",
                  available_keys.status_code)
            print("[DEBUG] Reason for failure:", available_keys.reason)
            sys.exit()

    def cos_key_check(self, instance_guid): # noqa
        """
        This function:
        1. Retrieves list of keys under
           provided instance guid.
        2. Prompts user choice to choose from existing
           keys or create new key
        :param instance_guid: guid of instance under which
               presence of keys need to be searched.
        :return: list of existing keys, guid of keys and user
               choice. Exit when error occurs in key retrieval request.
        """
        headers = {
            'Authorization': self.iam_access_token,
        }
        available_keys = requests.get("https://resource-controller."
                                      "cloud.ibm.com/v2/resource_keys",
                                      headers=headers)
        if available_keys.status_code == 200:
            available_keys = available_keys.json()
            # list for storing existing keys and list
            existing_keys = []
            existing_key_guid = []
            count = 0
            for k in available_keys['resources']:
                try:
                    # Check if provided instance guid matches with current
                    instance_id = k['resource_instance_url'].split('/')[-1]
                    if instance_id == instance_guid:
                        count += 1
                        print("{:2d}. {}".format(count, k['name']))
                        existing_keys.append(k['name'])
                        existing_key_guid.append(k['guid'])
                except KeyError:
                    continue
            if len(existing_keys) > 0:
                print('{:2d}. {}'.format(int(count) + 1,
                                         '* Create New Service Credentials *'))
            else:
                print('{:2d}. {}'.format(1,
                                         '* Create New Service Credentials *'))
            existing_keys.append('Create New Key')
            # Prompt for user input
            if len(existing_keys) > 0:
                while True:
                    key_option = input("[PROMPT] Your selection:  ").strip()
                    if not key_option.isdigit() or \
                       int(key_option) < 1 or \
                       int(key_option) >= (len(existing_keys) + 1):
                        print("[MESSAGE] Enter a number between 1 and "
                              "{}.".format(len(existing_keys)))
                        continue
                    else:
                        return existing_keys, key_option, \
                               existing_key_guid
            else:
                existing_keys = ['Create New Key']
                key_option = '1'
                existing_key_guid = []
                return existing_keys, key_option, \
                    existing_key_guid
        else:
            print('[DEBUG] key not present')
            print("[DEBUG] Failing with status code:",
                  available_keys.status_code)
            print("[DEBUG] Reason for failure:",
                  available_keys.reason)
            sys.exit()
