tmux new-session -d -s waterbirds_session -n device0_window
tmux send-keys -t waterbirds_session:device0_window.0 "python waterbirds_spare.py --pretrained=True --infer_lr=0.001 --infer_weight_decay=0.01 --infer_num_epochs=1 --batch_size=64 --device 0" C-m

tmux new-window -t waterbirds_session -n device0_window.1
tmux send-keys -t waterbirds_session:device0_window.1 "python waterbirds_spare.py --pretrained=True --infer_lr=0.0001 --infer_weight_decay=0.01 --infer_num_epochs=1 --batch_size=64 --device 0" C-m

tmux new-window -t waterbirds_session -n device0_window.2
tmux send-keys -t waterbirds_session:device0_window.2 "python waterbirds_spare.py --pretrained=True --infer_lr=1e-05 --infer_weight_decay=0.01 --infer_num_epochs=1 --batch_size=64 --device 0" C-m

tmux new-window -t waterbirds_session -n device0_window.3
tmux send-keys -t waterbirds_session:device0_window.3 "python waterbirds_spare.py --pretrained=True --infer_lr=0.001 --infer_weight_decay=0.1 --infer_num_epochs=1 --batch_size=64 --device 0" C-m

tmux new-window -t waterbirds_session -n device0_window.4
tmux send-keys -t waterbirds_session:device0_window.4 "python waterbirds_spare.py --pretrained=True --infer_lr=0.0001 --infer_weight_decay=0.1 --infer_num_epochs=1 --batch_size=64 --device 0" C-m

tmux new-window -t waterbirds_session -n device1_window
tmux send-keys -t waterbirds_session:device1_window.0 "python waterbirds_spare.py --pretrained=True --infer_lr=1e-05 --infer_weight_decay=0.1 --infer_num_epochs=1 --batch_size=64 --device 1" C-m

tmux new-window -t waterbirds_session -n device1_window.1
tmux send-keys -t waterbirds_session:device1_window.1 "python waterbirds_spare.py --pretrained=True --infer_lr=0.001 --infer_weight_decay=1 --infer_num_epochs=1 --batch_size=64 --device 1" C-m

tmux new-window -t waterbirds_session -n device1_window.2
tmux send-keys -t waterbirds_session:device1_window.2 "python waterbirds_spare.py --pretrained=True --infer_lr=0.0001 --infer_weight_decay=1 --infer_num_epochs=1 --batch_size=64 --device 1" C-m

tmux new-window -t waterbirds_session -n device1_window.3
tmux send-keys -t waterbirds_session:device1_window.3 "python waterbirds_spare.py --pretrained=True --infer_lr=1e-05 --infer_weight_decay=1 --infer_num_epochs=1 --batch_size=64 --device 1" C-m

tmux new-window -t waterbirds_session -n device2_window
tmux send-keys -t waterbirds_session:device2_window.0 "python waterbirds_spare.py --pretrained=True --infer_lr=0.001 --infer_weight_decay=0.01 --infer_num_epochs=2 --batch_size=64 --device 2" C-m

tmux new-window -t waterbirds_session -n device2_window.1
tmux send-keys -t waterbirds_session:device2_window.1 "python waterbirds_spare.py --pretrained=True --infer_lr=0.0001 --infer_weight_decay=0.01 --infer_num_epochs=2 --batch_size=64 --device 2" C-m

tmux new-window -t waterbirds_session -n device2_window.2
tmux send-keys -t waterbirds_session:device2_window.2 "python waterbirds_spare.py --pretrained=True --infer_lr=1e-05 --infer_weight_decay=0.01 --infer_num_epochs=2 --batch_size=64 --device 2" C-m

tmux new-window -t waterbirds_session -n device2_window.3
tmux send-keys -t waterbirds_session:device2_window.3 "python waterbirds_spare.py --pretrained=True --infer_lr=0.001 --infer_weight_decay=0.1 --infer_num_epochs=2 --batch_size=64 --device 2" C-m

tmux new-window -t waterbirds_session -n device2_window.4
tmux send-keys -t waterbirds_session:device2_window.4 "python waterbirds_spare.py --pretrained=True --infer_lr=0.0001 --infer_weight_decay=0.1 --infer_num_epochs=2 --batch_size=64 --device 2" C-m

tmux new-window -t waterbirds_session -n device3_window
tmux send-keys -t waterbirds_session:device3_window.0 "python waterbirds_spare.py --pretrained=True --infer_lr=1e-05 --infer_weight_decay=0.1 --infer_num_epochs=2 --batch_size=64 --device 3" C-m

tmux new-window -t waterbirds_session -n device3_window.1
tmux send-keys -t waterbirds_session:device3_window.1 "python waterbirds_spare.py --pretrained=True --infer_lr=0.001 --infer_weight_decay=1 --infer_num_epochs=2 --batch_size=64 --device 3" C-m

tmux new-window -t waterbirds_session -n device3_window.2
tmux send-keys -t waterbirds_session:device3_window.2 "python waterbirds_spare.py --pretrained=True --infer_lr=0.0001 --infer_weight_decay=1 --infer_num_epochs=2 --batch_size=64 --device 3" C-m

tmux new-window -t waterbirds_session -n device3_window.3
tmux send-keys -t waterbirds_session:device3_window.3 "python waterbirds_spare.py --pretrained=True --infer_lr=1e-05 --infer_weight_decay=1 --infer_num_epochs=2 --batch_size=64 --device 3" C-m

tmux attach -t waterbirds_session
