'''
Created on 25. apr. 2018

@author: mmpe
'''
import os
import numpy as np

def read_lib(filename):
    with open(filename) as fid:
        lines = fid.readlines()

    descriptor = lines[0]
    nRoughnessClasses, nHeights, nSectorslib = map(int, lines[2].split())
    z0ref_lst = list(map(float, lines[4].split()))
    
    # TODO: Implement for specified z0 and height

#
#
#     for i:=0 to nRoughnessClasses-1 do read(fil,z0reflib[i]);
#       readln(fil);
#       for i:=0 to nHeights-1 do read(fil,zreflib[i]);
#       readln(fil);
#       for k:=0 to nRoughnessClasses-1 do
#       begin
#         for i:=0 to nSectorslib-1 do
#         begin
#           read(fil,freq[k,i]);
#           freq[k,i]:=freq[k,i]/100;
#         end;
#         readln(fil);
#         for i:=0 to nHeights-1 do
#         begin
#           for j:=0 to nSectorslib-1 do read(fil,WAlib[k,i,j]);
#           readln(fil);
#           for j:=0 to nSectorslib-1 do read(fil,Wklib[k,i,j]);
#           readln(fil);
#         end;
#       end;
    f,A,k = [np.array(lines[i].split()[:nSectorslib], dtype=np.float) for i in [8, 10, 12]]
    return f/100,A,k


if __name__ == '__main__':
    print(read_lib(os.path.dirname(__file__) + "/Colonel/LUT/Farms/Horns Rev 1/hornsrev2.lib"))
