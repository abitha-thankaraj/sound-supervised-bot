import urx #Modified version of python urx

class Robot(urx.Robot):
    """URScript lookup - https://s3-eu-west-1.amazonaws.com/ur-support-site/29983/Script%20command%20Examples.pdf"""
    
    def __init__(self, ip = "192.168.1.14", payload = 0.2):
        super().__init__(ip)
        self.set_payload(payload)
    
    def prepare_msg_speedj(self, trajectory, acc=0.01, t=0.1): 
        """Generate URScript message."""
        
        header = "def myProg():\n"
        end = "end\n"
        prog = header
                
        prog+="\tsync()\n"
        for idx, j_vels in enumerate(trajectory):
            j_vels = [round(i, self.max_float_length) for i in j_vels]            
            # Ref: https://s3-eu-west-1.amazonaws.com/ur-support-site/115824/scriptManual_SW5.11.pdf
            if t is None:
                 prog+="\tspeedj([{},{},{},{},{},{}], {})\n".format(*j_vels ,acc)
            else:
                prog+="\tspeedj([{},{},{},{},{},{}], {}, {})\n".format(*j_vels ,acc, t)

        prog+="\tstopj(%s)\n" % (acc*0.8) #Decelerate 
        prog += end
        
        print(prog)
        return prog