from models.models import create_model
from options.train_options import TrainOptions
from data.DOTA_dataset import DOTADatasetFolder
from torch.utils.data import DataLoader

train_data = DOTADatasetFolder()
train_dataloader = DataLoader(dataset=train_data, num_workers=0, batch_size=1, shuffle=False)

opt = TrainOptions().parse()
model = create_model(opt)
total_steps = 0
file = r'G:\PythonProject\Deblur_RemoteSensing\loss_data\loss.txt'
file = open(file, 'a+')
for epoch in range(2):
    for i, data in enumerate(train_dataloader):
        model.set_input(data)
        model.optimize_parameters()

        errors = model.get_current_errors()
        print(i, ':', errors)
        loss_error = map(str, list(errors.values()))
        file.write(' '.join(loss_error) + '\n')
    model.save('latest')

file.close()
