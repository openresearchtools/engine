#[cfg(any(target_os = "windows", target_os = "macos"))]
mod imp {
    use crate::app_icon::build_tray_icon_rgba;
    use anyhow::{Context, Result};
    use egui::Context as EguiContext;
    use std::sync::mpsc::{self, Receiver};
    use tray_icon::menu::{Menu, MenuEvent, MenuItem, PredefinedMenuItem};
    use tray_icon::{Icon, TrayIcon, TrayIconBuilder};

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum TrayAction {
        OpenController,
        RefreshCluster,
        QuitController,
    }

    pub struct ControllerTray {
        _tray_icon: TrayIcon,
        action_rx: Receiver<TrayAction>,
    }

    impl ControllerTray {
        pub fn new(egui_ctx: &EguiContext) -> Result<Self> {
            let menu = Menu::new();
            let open_item = MenuItem::new("Open Controller", true, None);
            let refresh_item = MenuItem::new("Refresh Cluster", true, None);
            let quit_item = MenuItem::new("Quit Controller", true, None);
            let (action_tx, action_rx) = mpsc::channel();

            menu.append(&open_item)
                .context("failed to append tray open item")?;
            menu.append(&refresh_item)
                .context("failed to append tray refresh item")?;
            menu.append(&PredefinedMenuItem::separator())
                .context("failed to append tray separator")?;
            menu.append(&quit_item)
                .context("failed to append tray quit item")?;

            let tray_icon = TrayIconBuilder::new()
                .with_tooltip("Engine")
                .with_menu(Box::new(menu))
                .with_icon(build_icon()?)
                .build()
                .context("failed to create tray icon")?;

            let open_id = open_item.id().clone();
            let refresh_id = refresh_item.id().clone();
            let quit_id = quit_item.id().clone();
            let repaint_ctx = egui_ctx.clone();
            let handler_open_id = open_id.clone();
            let handler_refresh_id = refresh_id.clone();
            let handler_quit_id = quit_id.clone();
            MenuEvent::set_event_handler(Some(move |event: MenuEvent| {
                let action = if event.id == handler_open_id {
                    Some(TrayAction::OpenController)
                } else if event.id == handler_refresh_id {
                    Some(TrayAction::RefreshCluster)
                } else if event.id == handler_quit_id {
                    Some(TrayAction::QuitController)
                } else {
                    None
                };

                if let Some(action) = action {
                    let _ = action_tx.send(action);
                    repaint_ctx.request_repaint();
                }
            }));

            Ok(Self {
                _tray_icon: tray_icon,
                action_rx,
            })
        }

        pub fn poll_actions(&self) -> Vec<TrayAction> {
            let mut actions = Vec::new();
            while let Ok(action) = self.action_rx.try_recv() {
                actions.push(action);
            }
            actions
        }
    }

    fn build_icon() -> Result<Icon> {
        let (rgba, width, height) = build_tray_icon_rgba()?;
        Icon::from_rgba(rgba, width, height).context("failed to build tray icon pixels")
    }
}

#[cfg(not(any(target_os = "windows", target_os = "macos")))]
mod imp {
    use anyhow::{bail, Result};

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum TrayAction {
        OpenController,
        RefreshCluster,
        QuitController,
    }

    pub struct ControllerTray;

    impl ControllerTray {
        pub fn new(_egui_ctx: &egui::Context) -> Result<Self> {
            bail!("system tray is not implemented on this platform")
        }

        pub fn poll_actions(&self) -> Vec<TrayAction> {
            Vec::new()
        }
    }
}

pub use imp::{ControllerTray, TrayAction};
