import PySimpleGUI as sg
import os
import time

# sg.theme('LightBlue')

class App():
    def __init__(self, dir=os.getcwd()):
        self.dir = os.path.join(dir)
        font = ('Arial',20)

        step1 = [
            [
                sg.Text("(1) Tokenize", font=font),
            ],
            [
                sg.Listbox(values=self.get_vcf_files(), size=(20,21), key='VCF LIST', enable_events=True, font=font),
            ],
            [
                sg.Button('Transform VCF to Token', key="V2T", font=font),
            ],
        ]
        
        step2 = [
            [
                sg.Text("(2) Conversion", font=font),
            ],
            [
                sg.Listbox(values=self.get_token_files(), size=(20,21), key='TOKEN LIST', enable_events=True, font=font),
            ],
            [
                sg.Button('Transform Tokens to CVs', key="T2CV", font=font),
            ],
        ]
        
        step3 = [
            [
                sg.Text("(3) Diagnosis", font=font),
                sg.DropDown(values=self.get_cv_files(), key="CV LIST", enable_events=True, font=font)
            ],
            [
                sg.Image(
                    source=None,
                    key="ATTN MAP",
                    size=(800, 510),
                )
            ],
            [
                sg.Text(f"Risk Score:", font=font),
                sg.Text(f"-.--", key="RISK SCORE", font=font),
            ],
        ]

        self.layout = [
            [
                sg.Text(f"{self.dir}", font=font),
                sg.Button('Change Directory', key="CD", font=font),
            ],
            [
                sg.Column(step1), 
                sg.Column(step2), 
                sg.Column(step3),
            ],
            [
                sg.Text(f"     Progress", font=font),
                sg.Graph(
                    canvas_size=(1100, 30),
                    graph_bottom_left=(0, 0),
                    graph_top_right=(1100, 30),
                    key="PROGRESS BAR",
                    background_color="white",
                ),
                sg.Text(f"0", key="PROGRESS", font=font),
                sg.Text(f"%", font=font),
            ],
        ]
        
        self.window = sg.Window("CoVA", self.layout, finalize=True)
        self.window.bind('<Motion>', 'Motion')
        self.window.bind("<KeyPress>", "KeyPress")
        self.attn_map = self.window["ATTN MAP"]
        self.attn_map.update("attn_map_brank.png")
        self.risk_score = self.window["RISK SCORE"]
        self.progress_bar = self.window["PROGRESS BAR"]
        self.progress = self.window["PROGRESS"]
        
        self.selected_vcf = None
        self.selected_token = None
        self.progress_value = None
        self.running_event = None
        
    
    def get_vcf_files(self):
        return [f for f in os.listdir(self.dir) if f.endswith('.vcf.gz')]
    
    def get_token_files(self):
        return [f for f in os.listdir(self.dir) if f.endswith('.tokens.pt')]
    
    def get_cv_files(self):
        return [f for f in os.listdir(self.dir) if f.endswith('.cv.pt')]
    
    def restart_progress(self, event):
        self.progress_value = 0
        self.progress_bar.erase()
        self.running_event = event
    
    def add_progress(self):
        i = self.progress_value
        self.progress_bar.draw_rectangle((11*(i-1),0),(11*i-2,30),"black")
    
    def run(self):
        while True:
            event, values = self.window.read()
            
            if event == sg.WIN_CLOSED:
                break
            
            if event == "VCF LIST":
                for file_name in values[event]:
                    file_path = os.path.join(self.dir, file_name)
                    self.selected_vcf = file_path
            
            if event == "V2T":
                print(self.selected_vcf)
                self.restart_progress(event)
                
            if event == "TOKEN LIST":
                for file_name in values[event]:
                    file_path = os.path.join(self.dir, file_name)
                    self.selected_token = file_path
            
            if event == "T2CV":
                print(self.selected_token)
                self.restart_progress(event)
                
            if event == "CV LIST":
                file_path = os.path.join(self.dir, values[event])
                self.selected_cv = file_path
                self.restart_progress(event)
                
            if self.running_event == "CV LIST" and self.progress_value is None:
                self.attn_map.update("attn_map_sample.png")
                self.risk_score.update("0.99")
            
            if self.progress_value is not None:
                self.progress_value += 1
                self.add_progress()
                self.progress.update(str(self.progress_value))
                if self.progress_value == 100:
                    self.progress_value = None
                
        self.window.close()
        
if __name__ == "__main__":
    app = App()
    app.run()