from control_gen import ControlGen
import argparse
    
parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', type=str, required=True, help='Model Name')
parser.add_argument('--api', type=str, required=True, help='Api Name', choices=["get_yz", "get_yz_templated",  "get_z", "transfer_style"])
parser.add_argument('--template_id', type=int, help='id of template', default=-1, choices=[0,1,2,3,4,5,6,7,8,9,10])
parser.add_argument('--style_id', type=int, help='id of style', default=-1, choices=[0,1,2,3,4,5,6,7])
parser.add_argument('--device', default="cuda", help='device for model')
args = parser.parse_args()

# make sure template is given
if args.api == "get_yz_templated":
    assert args.template_id != -1
if args.api == "transfer_style":
    assert args.style_id != -1
    
# load model
model = ControlGen(model_path = args.model, 
                      device=args.device)

x = (13, 6, 2020)
x0 = (22, 6, 2020)
x1 = (7, 12, 2010)
x2 = (26, 7, 2001)
x3 = (31, 8, 2005)
template_list = ['JJA+CDI.',
                 'JJCA+EDI.',
                 'JJA+C..+DI.',
                 'JJCA+..+DI.',
                 'JJHB+.CEDI.',
                 'JJCHB+EDI.',
                 'JJCHB+..+DI.',
                 'JJHB+.C..+DI.',
                 '[it][is][the]B+[of]C[.].',
                 '[the][year][is]D[.].',
                 '.*A+CD..']

y = ['the', 'date', 'is', 'the', 'thirteen', 'of', 'june', ',', '2020', '.']
y_list = [['today', 'is', 'twenty', 'two', 'june', '2020', '.'],
          ['today', 'is', 'june', 'twenty', 'two', ',', '2020', '.'],
          ['today', 'is', 'twenty', 'two', 'june', 'of', 'the','year','2020', '.'],
          ['the', 'date', 'is', 'june', 'twenty', 'two', 'in', 'the','year','2020', '.'],
          ['it', 'is', 'the', 'twenty', 'second', 'of', 'june', ',', '2000', '.'],
          ['today', 'is', 'june', 'the', 'twenty', 'second', ',', '2020', '.'],
          ['it', 'is', 'june', 'the', 'twenty', 'second', 'of','the','year','2020', '.'],
          ['the', 'date','is', 'the', 'twenty', 'second', 'of', 'june', 'in', 'the','year','2020','.']]

print("Testing " + args.api + " for " + args.model)
print('--'*30)

if args.api == "get_yz":
    out = model.get_yz(x)
    print("x: " +  " ".join([str(el) for el in x]))
    print()
    print("y: " + " ".join(out["y"]))
    print()
    print("z: " + " ".join([str(el) for el in out["z"]]))
elif args.api == "get_yz_templated":
    template = template_list[args.template_id]
    out = model.get_yz(x, template)
    print("x: " +  " ".join([str(el) for el in x]))
    print()
    print("y: " + " ".join(out["y"]))
    print()
    print("z template: " + template)
    print()
    print("z: " + " ".join([str(el) for el in out["z"]]))   
elif args.api == "get_z":
    z = model.get_z(x, y)
    print("x: " +  " ".join([str(el) for el in x]))
    print()
    print("y: " + " ".join(y))
    print()
    print("z: " + " ".join([str(el) for el in z]))  
elif args.api == "transfer_style":
    print("Transfering style from : " + " ".join(y_list[args.style_id]))
    print(x1, " ".join(model.transfer_style(x0, y_list[args.style_id], x1)))
    print(x2, " ".join(model.transfer_style(x0, y_list[args.style_id], x2)))
    print(x3, " ".join(model.transfer_style(x0, y_list[args.style_id], x3)))
    print()
else:
    raise ValueError("Should not reach here")