import model
import numpy as np
from bokeh.plotting import figure
from bokeh.layouts import gridplot, column, row
from bokeh.models import ColumnDataSource, Button, Arrow, NormalHead, Div, Range1d, Spinner, Spacer, Patch
from bokeh.events import ButtonClick
from bokeh.application import Application
from bokeh.server.server import Server
from bokeh.application.handlers import FunctionHandler

config = {
    "epochs": 500,
    "learning_rate": 0.2,
    "init_theta": 1,
    "init_psi": 1,
    "plot_interval": (-2, 2),
    "update_interval_ms": 500,
    "gan_params": {
        "wgan_clip": 1,
        "wgp_target": 0.3,
        "inst_noise_std": 0.7,
        "gp_reg": 0.3,
        "co_reg": 1
    }
}
train_results = None  

def quiver(self, X, Y, U, V, color, alpha=1):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            cx, cy, cu, cv = X[i, j], Y[i, j], U[i, j], V[i, j]
            d = np.sqrt(cu**2 + cv**2) 
            head_size = min(4, 4 * (d/0.1))
            nh = NormalHead(fill_color=color, fill_alpha=alpha, line_color=color, size=head_size)
            self.add_layout(Arrow(end=nh, line_color=color, line_width=2, line_alpha=alpha, level="underlay",
                   x_start=cx, y_start=cy, x_end=cx+cu, y_end=cy+cv))

def train():
    # standard gan:
    gan = model.Model(iterations=config["epochs"], learning_rate=config["learning_rate"])

    # NGAN:
    ngan = model.Model(iterations=config["epochs"], learning_rate=config["learning_rate"])
    ngan.set_loss('NGAN')

    # WGAN:
    wgan = model.Model(iterations=config["epochs"], learning_rate=config["learning_rate"])
    wgan.set_loss('WGAN', config["gan_params"]["wgan_clip"])

    # WGAN-GP:
    wgan_gp = model.Model(iterations=config["epochs"], learning_rate=config["learning_rate"])
    wgan_gp.set_loss('WGAN')
    wgan_gp.set_regularization_loss('WGP', gamma_gp=config["gan_params"]["wgp_target"])

    # IGP:
    igp = model.Model(iterations=config["epochs"], learning_rate=config["learning_rate"])
    igp.set_instance_noise(True, config["gan_params"]["inst_noise_std"])

    # GP:
    gp = model.Model(iterations=config["epochs"], learning_rate=config["learning_rate"])
    gp.set_regularization_loss('GP', gamma_gp=config["gan_params"]["gp_reg"])

    # CRGP:
    crgp = model.Model(iterations=config["epochs"], learning_rate=config["learning_rate"])
    crgp.set_regularization_loss('CRGP')

    # CO:
    co = model.Model(iterations=config["epochs"], learning_rate=config["learning_rate"])
    co.set_regularization_loss('CO', gamma_gp=config["gan_params"]["co_reg"])

    # DRAGAN:
    dragan = model.Model(iterations=config["epochs"], learning_rate=config["learning_rate"])
    dragan.set_regularization_loss('DRAGAN')
    dragan.set_instance_noise(True)

    # Simple LeCam_Regularization:
    simple_lecam_reg = model.Model(iterations=config["epochs"], learning_rate=config["learning_rate"])
    simple_lecam_reg.set_regularization_loss('SimpleLeCam')

    # LeCam as in paper: "Regularizing Generative Adversarial Networks under Limited Data"
    lecam_reg = model.Model(iterations=config["epochs"], learning_rate=config["learning_rate"])
    lecam_reg.set_regularization_loss('LeCam')

    training_algorithms = [
        (gan, "Standard GAN"),
        (ngan, "Non-saturating GAN"),
        (wgan, "Wasserstein GAN"),
        (wgan_gp, "Wasserstein GAN with GP"),
        (igp, "Instance Noise GAN"),
        (gp, "GAN with Gradient Penalty"),
        (crgp, "Critically Penalized GAN"),
        (co,"Consensus Optimization"),
        (dragan, "DRAGAN Gradient Penalty"),
        (simple_lecam_reg, "LeCam-Distance as regularization"),
        (lecam_reg, "LeCam GAN")
    ]

    train_results = {}
    for gan_model, name in training_algorithms:
        init_theta = config["init_theta"]
        init_psi = config["init_psi"]
        if name == "Wasserstein GAN": 
            clip = config["gan_params"]["wgan_clip"]
            init_psi = np.clip(init_psi, -clip, clip).item()
        thetas, psis = gan_model.train(init_theta=init_theta, init_psi=init_psi)
        X, Y, U, V,vectors = gan_model.get_vectors(interval=config["plot_interval"], steps=15)
        train_results[name] = (gan_model, thetas, psis, X, Y, U, V) 

    return train_results 

def make_plots(train_results):
    plots = []
    sources = {}
    start_sources = {}
    for name, (gan_model, thetas, psis, X, Y, U, V) in train_results.items():
        init_theta = config["init_theta"]
        init_psi = config["init_psi"]
        if name == "Wasserstein GAN": 
            clip = config["gan_params"]["wgan_clip"]
            init_psi = np.clip(init_psi, -clip, clip).item()
        rmin, rmax = config["plot_interval"]
        x_range = Range1d(rmin, rmax, bounds=(rmin, rmax))
        y_range = Range1d(rmin, rmax, bounds=(rmin, rmax))
        p = figure(title=name, x_axis_label='θ', y_axis_label='Ψ', tools="pan,wheel_zoom,box_zoom,reset,save", 
                x_range=x_range, y_range=y_range)
        #quiver(p, X, Y, U, V, color="black")
        # ^^^ use this for more control (but it's slower)

        cds = ColumnDataSource(data=dict(x_start=X, y_start=Y, x_end=X+U, y_end=Y+V))
        p.add_layout(Arrow(end=NormalHead(line_color="black", fill_color="black", size=4), source=cds, 
                        x_start='x_start', y_start='y_start', x_end='x_end', y_end='y_end', 
                        line_width=2, level="underlay", line_color="black"))

        source = ColumnDataSource(data=dict(x=[], y=[]))
        source_start = ColumnDataSource(data=dict(x=[], y=[]))
        p.scatter("x", "y", source=source, color="blue", size=8)
        p.scatter("x", "y", source=source_start, color="red", size=8)

        if name == "Wasserstein GAN":
            clamp = gan_model.clamp
            x_shade = [rmin, rmax, rmax, rmin]
            y_shade = [rmax, rmax, clamp, clamp]
            wgan_patch_source_top = ColumnDataSource(data=dict(x=x_shade, y=y_shade))
            x_shade2 = [rmin, rmax, rmax, rmin]
            y_shade2 = [-clamp, -clamp, rmin, rmin]
            wgan_patch_source_bottom = ColumnDataSource(data=dict(x=x_shade2, y=y_shade2))

            p.patch("x", "y", source=wgan_patch_source_top, color="grey", alpha=0.5, line_width=0)
            p.patch("x", "y", source=wgan_patch_source_bottom, color="grey", alpha=0.5, line_width=0)

            source_start.data = dict(x=[init_theta], y=[init_psi])
        else:
            source_start.data = dict(x=[init_theta], y=[init_psi])
            
        sources[name] = (source, thetas, psis)
        start_sources[name] = source_start
        plots.append(p) 

    return plots, sources, start_sources, [wgan_patch_source_top, wgan_patch_source_bottom]

def ui(doc, plots, sources, start_sources, wgan_patch_sources):
    frame = 0
    paused = False 
    def get_update(sources, start_sources):
        def update():
            nonlocal frame, paused 
            if paused: return
            reset = False 
            if frame >= config["epochs"]:
                reset = True 
                frame = 0 
            
            for name, source in start_sources.items():
                init_theta = config["init_theta"]
                init_psi = config["init_psi"]
                if name == "Wasserstein GAN":
                    clip = config["gan_params"]["wgan_clip"]
                    init_psi = np.clip(init_psi, -clip, clip).item()
                source.data = dict(x=[init_theta], y=[init_psi])
            
            for source, thetas, psis in sources.values():
                if reset: source.data = dict(x=[], y=[])
                else: source.stream(dict(x=[thetas[frame]], y=[psis[frame]]), rollover=config["epochs"])
            if not reset: frame += 1 
        return update 
    
    grid = gridplot(plots, ncols=3, width=300, height=300, toolbar_options=dict(logo=None))
    frame = 0
    doc.add_periodic_callback(get_update(sources, start_sources), config["update_interval_ms"])

    # Buttons & final layout 
    header = Div(text="<h1>DiracGAN</h1>")
    footer = Div(text="<h4 style='color:gray'>Deep Generative Models 2024 - Project by Emanuele Harrasser, Milan Binz, Julian Bayer</h4>")

    anim_controls = Div(text="<h2>Animation Controls</h2>")
    reset_btn = Button(label="Reset Animation")
    pause_btn = Button(label="Pause Animation")
    skip_to_end_btn = Button(label="Skip to end") 

    def reset(sources, start_sources):
        def reset_action(_):
            nonlocal frame 
            frame = 0
            for source, _, _ in sources.values():
                source.data = dict(x=[], y=[])

            for name, source in start_sources.items():
                init_theta = config["init_theta"]
                init_psi = config["init_psi"]
                if name == "Wasserstein GAN":
                    clip = config["gan_params"]["wgan_clip"]
                    init_psi = np.clip(init_psi, -clip, clip).item()
                source.data = dict(x=[init_theta], y=[init_psi])

        return reset_action

    def toggle_paused(): 
        def pause_action(_):
            pause_btn.label = ["Pause Animation", "Resume Animation"][["Resume Animation", "Pause Animation"].index(pause_btn.label)]
            nonlocal paused
            paused = not paused
        return pause_action

    def skip_to_end(sources):
        def skip_to_end_action(_):
            nonlocal frame 
            if not paused: toggle_paused()(_) 
            for source, thetas, psis in sources.values():
                source.data = dict(x=thetas, y=psis)
            for name, source in start_sources.items():
                init_theta = config["init_theta"]
                init_psi = config["init_psi"]
                if name == "Wasserstein GAN":
                    clip = config["gan_params"]["wgan_clip"]
                    init_psi = np.clip(init_psi, -clip, clip).item()
                source.data = dict(x=[init_theta], y=[init_psi])
            frame = config["epochs"]
        return skip_to_end_action

    reset_btn.on_event(ButtonClick, reset(sources, start_sources))
    pause_btn.on_event(ButtonClick, toggle_paused())
    skip_to_end_btn.on_event(ButtonClick, skip_to_end(sources))
    anim_buttons = row(reset_btn, pause_btn, skip_to_end_btn)

    param_controls = Div(text="<h2>Parameters</h2>")
    retrain_button = Button(label="Retrain") 
    spinner_epochs = Spinner(title="Epochs", low=1, high=10000, step=1, value=config["epochs"])
    spinner_lr = Spinner(title="Learning Rate", low=1e-6, high=1, step=0.01, value=config["learning_rate"])
    rmin, rmax = config["plot_interval"]
    spinner_start_x = Spinner(title="Initial θ", low=rmin, high=rmax, step=0.01, value=config["init_theta"])
    spinner_start_y = Spinner(title="Initital Ψ", low=rmin, high=rmax, step=0.01, value=config["init_psi"])
    spinner_wgan_clip = Spinner(title="WGAN Clip", low=rmin, high=rmax, step=0.01, value=config["gan_params"]["wgan_clip"])
    spinner_wgp_reg  = Spinner(title="WGP Reg", low=0, high=1, step=0.01, value=config["gan_params"]["wgp_target"])
    spinner_inst_noise_std = Spinner(title="Inst. Noise STD", low=0, high=1, step=0.01, value=config["gan_params"]["inst_noise_std"])
    spinner_gp_reg  = Spinner(title="GP Reg", low=0, high=1, step=0.01, value=config["gan_params"]["gp_reg"])
    spinner_co_reg = Spinner(title="CO Reg", low=0, high=1, step=0.01, value=config["gan_params"]["co_reg"])

    def retrain(): 
        def retrain_action(_):
            global train_results
            nonlocal sources 
            config["epochs"] = spinner_epochs.value 
            config["learning_rate"] = spinner_lr.value 
            config["init_theta"] = spinner_start_x.value             
            config["init_psi"] = spinner_start_y.value     
            config["gan_params"]["wgan_clip"] = spinner_wgan_clip.value
            config["gan_params"]["wgp_target"] = spinner_wgp_reg.value
            config["gan_params"]["inst_noise_std"] = spinner_inst_noise_std.value
            config["gan_params"]["gp_reg"] = spinner_gp_reg.value
            config["gan_params"]["co_reg"] = spinner_co_reg.value

            train_results = train() 
            for name, _ in sources.items():
                sources[name] = (sources[name][0], train_results[name][1], train_results[name][2])
            
            # update wgan patch 
            wpst, wpsb = wgan_patch_sources
            clamp = train_results["Wasserstein GAN"][0].clamp
            x_shade = [rmin, rmax, rmax, rmin]
            y_shade = [rmax, rmax, clamp, clamp]
            wpst.data = dict(x=x_shade, y=y_shade)
            x_shade2 = [rmin, rmax, rmax, rmin]
            y_shade2 = [-clamp, -clamp, rmin, rmin]
            wpsb.data = dict(x=x_shade2, y=y_shade2)

            reset(sources, start_sources)(_) 

        return retrain_action
    
    retrain_button.on_event(ButtonClick, retrain()) 
    param_buttons = column(column(Div(text="<h3>General</h3>"),
                                  row(spinner_epochs, spinner_lr), 
                                  row(spinner_start_x, spinner_start_y)),
                                  Div(text="<h3>Methods</h3>"),
                                  row(spinner_wgan_clip),
                                  row(spinner_wgp_reg),
                                  row(spinner_inst_noise_std),
                                  row(spinner_gp_reg),
                                  row(spinner_co_reg),
                                  retrain_button) 

    all_controls = column(anim_controls, anim_buttons, param_controls, param_buttons) 

    layout = column(header, row(grid, Spacer(width=25), all_controls), footer)
    return layout

def make_document(doc):
    while train_results is None: continue 
    plots, sources, start_sources, wgan_patch_sources = make_plots(train_results)
    root = ui(doc, plots, sources, start_sources, wgan_patch_sources)
    doc.add_root(root)
    doc.set_title("DiracGAN")

def setup():
    global train_results 
    print("Training models...")
    train_results = train() 
    print("Training finished.")
    print("Starting server...")

    app = Application(FunctionHandler(make_document))
    server = Server({'/': app})
    server.start()
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.add_callback(lambda: print(f"Server running on http://{'localhost' if server.address is None else server.address}:{server.port}"))
    server.io_loop.add_callback(lambda: print(f"Press CTRL + C to quit."))
    try:
        server.io_loop.start()
    except KeyboardInterrupt:
        server.io_loop.stop()
        server.stop()
        print("Quitting...")

if __name__ in "__main__": 
    setup() 