"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[7531],{1923:(e,n,t)=>{t.r(n),t.d(n,{assets:()=>l,contentTitle:()=>s,default:()=>d,frontMatter:()=>i,metadata:()=>o,toc:()=>c});var r=t(5893),a=t(1151);const i={title:"FSM Group Chat -- User-specified agent transitions",authors:["joshkyh","freedeaths"],tags:["AutoGen"]},s=void 0,o={permalink:"/autogen/blog/2024/02/11/FSM-GroupChat",source:"@site/blog/2024-02-11-FSM-GroupChat/index.mdx",title:"FSM Group Chat -- User-specified agent transitions",description:"FSM Group Chat",date:"2024-02-11T00:00:00.000Z",formattedDate:"February 11, 2024",tags:[{label:"AutoGen",permalink:"/autogen/blog/tags/auto-gen"}],readingTime:5.655,hasTruncateMarker:!1,authors:[{name:"Joshua Kim",title:"AI Freelancer at SpectData",url:"https://github.com/joshkyh/",imageURL:"https://github.com/joshkyh.png",key:"joshkyh"},{name:"Yishen Sun",title:"Data Scientist at PingCAP LAB",url:"https://github.com/freedeaths/",imageURL:"https://github.com/freedeaths.png",key:"freedeaths"}],frontMatter:{title:"FSM Group Chat -- User-specified agent transitions",authors:["joshkyh","freedeaths"],tags:["AutoGen"]},unlisted:!1,prevItem:{title:"What's New in AutoGen?",permalink:"/autogen/blog/2024/03/03/AutoGen-Update"},nextItem:{title:"Anny: Assisting AutoGen Devs Via AutoGen",permalink:"/autogen/blog/2024/02/02/AutoAnny"}},l={authorsImageUrls:[void 0,void 0]},c=[{value:"TL;DR",id:"tldr",level:2},{value:"Possible use-cases for transition graph",id:"possible-use-cases-for-transition-graph",level:2},{value:"Usage Guide",id:"usage-guide",level:2},{value:"Application of the FSM Feature",id:"application-of-the-fsm-feature",level:3},{value:"Usage",id:"usage",level:3},{value:"Notebook examples",id:"notebook-examples",level:2}];function h(e){const n={a:"a",code:"code",h2:"h2",h3:"h3",img:"img",li:"li",ol:"ol",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,a.a)(),...e.components};return(0,r.jsxs)(r.Fragment,{children:[(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.img,{alt:"FSM Group Chat",src:t(6940).Z+"",width:"828",height:"512"}),"\r\n",(0,r.jsx)("p",{align:"center",children:(0,r.jsx)("em",{children:"Finite State Machine (FSM) Group Chat allows the user to constrain agent transitions."})})]}),"\n",(0,r.jsx)(n.h2,{id:"tldr",children:"TL;DR"}),"\n",(0,r.jsx)(n.p,{children:"Recently, FSM Group Chat is released that allows the user to input a transition graph to constrain agent transitions. This is useful as the number of agents increases because the number of transition pairs (N choose 2 combinations) increases exponentially increasing the risk of sub-optimal transitions, which leads to wastage of tokens and/or poor outcomes."}),"\n",(0,r.jsx)(n.h2,{id:"possible-use-cases-for-transition-graph",children:"Possible use-cases for transition graph"}),"\n",(0,r.jsxs)(n.ol,{children:["\n",(0,r.jsx)(n.li,{children:"One-pass workflow, i.e., we want each agent to only have one pass at the problem, Agent A -> B -> C."}),"\n",(0,r.jsx)(n.li,{children:"Decision tree flow, like a decision tree, we start with a root node (agent), and flow down the decision tree with agents being nodes. For example, if the query is a SQL query, hand over to the SQL agent, else if the query is a RAG query, hand over to the RAG agent."}),"\n",(0,r.jsx)(n.li,{children:"Sequential Team Ops. Suppose we have a team of 3 developer agents, each responsible for a different GitHub repo. We also have a team of business analyst that discuss and debate the overall goal of the user. We could have the manager agent of the developer team speak to the manager agent of the business analysis team. That way, the discussions are more focused team-wise, and better outcomes can be expected."}),"\n"]}),"\n",(0,r.jsx)(n.p,{children:"Note that we are not enforcing a directed acyclic graph; the user can specify the graph to be acyclic, but cyclic workflows can also be useful to iteratively work on a problem, and layering additional analysis onto the solution."}),"\n",(0,r.jsx)(n.h2,{id:"usage-guide",children:"Usage Guide"}),"\n",(0,r.jsxs)(n.p,{children:["We have added two parameters ",(0,r.jsx)(n.code,{children:"allowed_or_disallowed_speaker_transitions"})," and ",(0,r.jsx)(n.code,{children:"speaker_transitions_type"}),"."]}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"allowed_or_disallowed_speaker_transitions"}),": is a dictionary with the type expectation of ",(0,r.jsx)(n.code,{children:"{Agent: [Agent]}"}),". The key refers to the source agent, while the value(s) in the list refers to the target agent(s). If none, a fully connection graph is assumed."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"speaker_transitions_type"}),': is a string with the type expectation of string, and specifically, one of ["allowed", "disallowed"]. We wanted the user to be able to supply a dictionary of allowed or disallowed transitions to improve the ease of use. In the code base, we would invert the disallowed transition into a allowed transition dictionary ',(0,r.jsx)(n.code,{children:"allowed_speaker_transitions_dict"}),"."]}),"\n"]}),"\n",(0,r.jsx)(n.h3,{id:"application-of-the-fsm-feature",children:"Application of the FSM Feature"}),"\n",(0,r.jsxs)(n.p,{children:["A quick demonstration of how to initiate a FSM-based ",(0,r.jsx)(n.code,{children:"GroupChat"})," in the ",(0,r.jsx)(n.code,{children:"AutoGen"})," framework. In this demonstration, if we consider each agent as a state, and each agent speaks according to certain conditions. For example, User always initiates the task first, followed by Planner creating a plan. Then Engineer and Executor work alternately, with Critic intervening when necessary, and after Critic, only Planner should revise additional plans. Each state can only exist at a time, and there are transition conditions between states. Therefore, GroupChat can be well abstracted as a Finite-State Machine (FSM)."]}),"\n",(0,r.jsx)(n.p,{children:(0,r.jsx)(n.img,{alt:"visualization",src:t(9408).Z+"",width:"816",height:"492"})}),"\n",(0,r.jsx)(n.h3,{id:"usage",children:"Usage"}),"\n",(0,r.jsxs)(n.ol,{start:"0",children:["\n",(0,r.jsx)(n.li,{children:"Pre-requisites"}),"\n"]}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-bash",children:"pip install autogen[graph]\n"})}),"\n",(0,r.jsxs)(n.ol,{children:["\n",(0,r.jsxs)(n.li,{children:["\n",(0,r.jsx)(n.p,{children:"Import dependencies"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-python",children:"from autogen.agentchat import GroupChat, AssistantAgent, UserProxyAgent, GroupChatManager\r\nfrom autogen.oai.openai_utils import config_list_from_dotenv\n"})}),"\n"]}),"\n",(0,r.jsxs)(n.li,{children:["\n",(0,r.jsx)(n.p,{children:"Configure LLM parameters"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-python",children:'# Please feel free to change it as you wish\r\nconfig_list = config_list_from_dotenv(\r\n        dotenv_file_path=\'.env\',\r\n        model_api_key_map={\'gpt-4-1106-preview\':\'OPENAI_API_KEY\'},\r\n        filter_dict={\r\n            "model": {\r\n                "gpt-4-1106-preview"\r\n            }\r\n        }\r\n    )\r\n\r\ngpt_config = {\r\n    "cache_seed": None,\r\n    "temperature": 0,\r\n    "config_list": config_list,\r\n    "timeout": 100,\r\n}\n'})}),"\n"]}),"\n",(0,r.jsxs)(n.li,{children:["\n",(0,r.jsx)(n.p,{children:"Define the task"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-python",children:'# describe the task\r\ntask = """Add 1 to the number output by the previous role. If the previous number is 20, output "TERMINATE"."""\n'})}),"\n"]}),"\n",(0,r.jsxs)(n.li,{children:["\n",(0,r.jsx)(n.p,{children:"Define agents"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-python",children:'# agents configuration\r\nengineer = AssistantAgent(\r\n    name="Engineer",\r\n    llm_config=gpt_config,\r\n    system_message=task,\r\n    description="""I am **ONLY** allowed to speak **immediately** after `Planner`, `Critic` and `Executor`.\r\nIf the last number mentioned by `Critic` is not a multiple of 5, the next speaker must be `Engineer`.\r\n"""\r\n)\r\n\r\nplanner = AssistantAgent(\r\n    name="Planner",\r\n    system_message=task,\r\n    llm_config=gpt_config,\r\n    description="""I am **ONLY** allowed to speak **immediately** after `User` or `Critic`.\r\nIf the last number mentioned by `Critic` is a multiple of 5, the next speaker must be `Planner`.\r\n"""\r\n)\r\n\r\nexecutor = AssistantAgent(\r\n    name="Executor",\r\n    system_message=task,\r\n    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("FINISH"),\r\n    llm_config=gpt_config,\r\n    description="""I am **ONLY** allowed to speak **immediately** after `Engineer`.\r\nIf the last number mentioned by `Engineer` is a multiple of 3, the next speaker can only be `Executor`.\r\n"""\r\n)\r\n\r\ncritic = AssistantAgent(\r\n    name="Critic",\r\n    system_message=task,\r\n    llm_config=gpt_config,\r\n    description="""I am **ONLY** allowed to speak **immediately** after `Engineer`.\r\nIf the last number mentioned by `Engineer` is not a multiple of 3, the next speaker can only be `Critic`.\r\n"""\r\n)\r\n\r\nuser_proxy = UserProxyAgent(\r\n    name="User",\r\n    system_message=task,\r\n    code_execution_config=False,\r\n    human_input_mode="NEVER",\r\n    llm_config=False,\r\n    description="""\r\nNever select me as a speaker.\r\n"""\r\n)\n'})}),"\n",(0,r.jsxs)(n.ol,{children:["\n",(0,r.jsxs)(n.li,{children:["Here, I have configured the ",(0,r.jsx)(n.code,{children:"system_messages"}),' as "task" because every agent should know what it needs to do. In this example, each agent has the same task, which is to count in sequence.']}),"\n",(0,r.jsx)(n.li,{children:(0,r.jsxs)(n.strong,{children:["The most important point is the ",(0,r.jsx)(n.code,{children:"description"})," parameter, where I have used natural language to describe the transition conditions of the FSM. Because the manager knows which agents are available next based on the constraints of the graph, I describe in the ",(0,r.jsx)(n.code,{children:"description"})," field of each candidate agent when it can speak, effectively describing the transition conditions in the FSM."]})}),"\n"]}),"\n"]}),"\n",(0,r.jsxs)(n.li,{children:["\n",(0,r.jsx)(n.p,{children:"Define the graph"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-python",children:"graph_dict = {}\r\ngraph_dict[user_proxy] = [planner]\r\ngraph_dict[planner] = [engineer]\r\ngraph_dict[engineer] = [critic, executor]\r\ngraph_dict[critic] = [engineer, planner]\r\ngraph_dict[executor] = [engineer]\n"})}),"\n",(0,r.jsxs)(n.ol,{children:["\n",(0,r.jsx)(n.li,{children:(0,r.jsx)(n.strong,{children:"The graph here and the transition conditions mentioned above together form a complete FSM. Both are essential and cannot be missing."})}),"\n",(0,r.jsx)(n.li,{children:"You can visualize it as you wish, which is shown as follows"}),"\n"]}),"\n",(0,r.jsx)(n.p,{children:(0,r.jsx)(n.img,{alt:"visualization",src:t(4149).Z+"",width:"1220",height:"1019"})}),"\n"]}),"\n",(0,r.jsxs)(n.li,{children:["\n",(0,r.jsxs)(n.p,{children:["Define a ",(0,r.jsx)(n.code,{children:"GroupChat"})," and a ",(0,r.jsx)(n.code,{children:"GroupChatManager"})]}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-python",children:'agents = [user_proxy, engineer, planner, executor, critic]\r\n\r\n# create the groupchat\r\ngroup_chat = GroupChat(agents=agents, messages=[], max_round=25, allowed_or_disallowed_speaker_transitions=graph_dict, allow_repeat_speaker=None, speaker_transitions_type="allowed")\r\n\r\n# create the manager\r\nmanager = GroupChatManager(\r\n    groupchat=group_chat,\r\n    llm_config=gpt_config,\r\n    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),\r\n    code_execution_config=False,\r\n)\n'})}),"\n"]}),"\n",(0,r.jsxs)(n.li,{children:["\n",(0,r.jsx)(n.p,{children:"Initiate the chat"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-python",children:'# initiate the task\r\nuser_proxy.initiate_chat(\r\n    manager,\r\n    message="1",\r\n    clear_history=True\r\n)\n'})}),"\n"]}),"\n",(0,r.jsxs)(n.li,{children:["\n",(0,r.jsx)(n.p,{children:"You may get the following output(I deleted the ignorable warning):"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{children:"User (to chat_manager):\r\n\r\n1\r\n\r\n--------------------------------------------------------------------------------\r\nPlanner (to chat_manager):\r\n\r\n2\r\n\r\n--------------------------------------------------------------------------------\r\nEngineer (to chat_manager):\r\n\r\n3\r\n\r\n--------------------------------------------------------------------------------\r\nExecutor (to chat_manager):\r\n\r\n4\r\n\r\n--------------------------------------------------------------------------------\r\nEngineer (to chat_manager):\r\n\r\n5\r\n\r\n--------------------------------------------------------------------------------\r\nCritic (to chat_manager):\r\n\r\n6\r\n\r\n--------------------------------------------------------------------------------\r\nEngineer (to chat_manager):\r\n\r\n7\r\n\r\n--------------------------------------------------------------------------------\r\nCritic (to chat_manager):\r\n\r\n8\r\n\r\n--------------------------------------------------------------------------------\r\nEngineer (to chat_manager):\r\n\r\n9\r\n\r\n--------------------------------------------------------------------------------\r\nExecutor (to chat_manager):\r\n\r\n10\r\n\r\n--------------------------------------------------------------------------------\r\nEngineer (to chat_manager):\r\n\r\n11\r\n\r\n--------------------------------------------------------------------------------\r\nCritic (to chat_manager):\r\n\r\n12\r\n\r\n--------------------------------------------------------------------------------\r\nEngineer (to chat_manager):\r\n\r\n13\r\n\r\n--------------------------------------------------------------------------------\r\nCritic (to chat_manager):\r\n\r\n14\r\n\r\n--------------------------------------------------------------------------------\r\nEngineer (to chat_manager):\r\n\r\n15\r\n\r\n--------------------------------------------------------------------------------\r\nExecutor (to chat_manager):\r\n\r\n16\r\n\r\n--------------------------------------------------------------------------------\r\nEngineer (to chat_manager):\r\n\r\n17\r\n\r\n--------------------------------------------------------------------------------\r\nCritic (to chat_manager):\r\n\r\n18\r\n\r\n--------------------------------------------------------------------------------\r\nEngineer (to chat_manager):\r\n\r\n19\r\n\r\n--------------------------------------------------------------------------------\r\nCritic (to chat_manager):\r\n\r\n20\r\n\r\n--------------------------------------------------------------------------------\r\nPlanner (to chat_manager):\r\n\r\nTERMINATE\n"})}),"\n"]}),"\n"]}),"\n",(0,r.jsx)(n.h2,{id:"notebook-examples",children:"Notebook examples"}),"\n",(0,r.jsxs)(n.p,{children:["More examples can be found in the ",(0,r.jsx)(n.a,{href:"https://microsoft.github.io/autogen/docs/notebooks/agentchat_groupchat_finite_state_machine/",children:"notebook"}),". The notebook includes more examples of possible transition paths such as (1) hub and spoke, (2) sequential team operations, and (3) think aloud and debate. It also uses the function ",(0,r.jsx)(n.code,{children:"visualize_speaker_transitions_dict"})," from ",(0,r.jsx)(n.code,{children:"autogen.graph_utils"})," to visualize the various graphs."]})]})}function d(e={}){const{wrapper:n}={...(0,a.a)(),...e.components};return n?(0,r.jsx)(n,{...e,children:(0,r.jsx)(h,{...e})}):h(e)}},9408:(e,n,t)=>{t.d(n,{Z:()=>r});const r=t.p+"assets/images/FSM_logic-99a03b9ae26fd5c532a53e5a9f250b14.png"},4149:(e,n,t)=>{t.d(n,{Z:()=>r});const r=t.p+"assets/images/FSM_of_multi-agents-a7b6f33a351cbf863c6f86466714afa3.png"},6940:(e,n,t)=>{t.d(n,{Z:()=>r});const r=t.p+"assets/images/teaser-3fe716ecdfd99422e6d735039400fa98.jpg"},1151:(e,n,t)=>{t.d(n,{Z:()=>o,a:()=>s});var r=t(7294);const a={},i=r.createContext(a);function s(e){const n=r.useContext(i);return r.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function o(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(a):e.components||a:s(e.components),r.createElement(i.Provider,{value:n},e.children)}}}]);