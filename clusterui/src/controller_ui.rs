use crate::agent::{available_public_api_bind_hosts, PublicApiBindHostOption};
use crate::catalog::{ManagedModelEntry, ManagedModelTask};
use crate::cluster_api::{DeviceInfo, ExecutionGroupInfo, RetentionMode};
use crate::model_metadata::{ModelFileMetadata, RuntimeVramEstimate};
use crate::model_store::{load_local_package_readme, supported_audio_repos};
use crate::protocol::{
    ClusterModelArtifactInfo, ClusterModelPackageInfo, LinkMetrics, NodeSnapshot,
    PairingRequestInfo, PlacementPlan, TelemetrySnapshot, CLUSTER_AGENT_RPC_PORT,
};
use crate::settings::ControllerThemePreference;
use crate::{
    format_bytes_compact, format_inference_metrics, format_link_metrics, format_mib,
    format_mib_from_bytes, instance_model_type_label as runtime_kind_label, labeled_f32,
    labeled_i32, placement_strategy_label, state_label, ClusterControllerApp, ModelStoreBusyState,
    ModelStoreMode,
};
use eframe::egui;
use egui_commonmark::CommonMarkViewer;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use std::time::Duration;
use time::{OffsetDateTime, UtcOffset};

const ENGINE_LICENSE_TEXT: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../LICENSE"));
const THIRD_PARTY_NOTICES_TEXT: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../third_party/README.md"
));
const THIRD_PARTY_LICENSES_TEXT: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../third_party/LICENSES.md"
));
const ENGINE_MANUAL_TEXT: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../docs/manual.md"));

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ControllerPage {
    Nodes,
    Instances,
    Models,
    Server,
    Settings,
    About,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum AboutDocument {
    EngineLicense,
    ThirdPartyNotices,
    ThirdPartyLicenses,
    Manual,
}

#[derive(Clone, Copy)]
struct ControllerPalette {
    canvas_fill: egui::Color32,
    separator: egui::Color32,
    card_fill: egui::Color32,
    outlined_card_fill: egui::Color32,
    border: egui::Color32,
    border_soft: egui::Color32,
    warning_fill: egui::Color32,
    warning_border: egui::Color32,
    warning_text: egui::Color32,
    muted_text: egui::Color32,
    accent_fill: egui::Color32,
    accent_stroke: egui::Color32,
    secondary_fill: egui::Color32,
    secondary_stroke: egui::Color32,
    nav_selected_fill: egui::Color32,
    nav_selected_stroke: egui::Color32,
    subtab_selected_fill: egui::Color32,
    subtab_selected_stroke: egui::Color32,
}

fn controller_palette(theme: egui::Theme) -> ControllerPalette {
    match theme {
        egui::Theme::Dark => ControllerPalette {
            canvas_fill: egui::Color32::from_rgb(30, 30, 30),
            separator: egui::Color32::from_rgb(58, 58, 58),
            card_fill: egui::Color32::from_rgb(37, 37, 38),
            outlined_card_fill: egui::Color32::from_rgb(45, 45, 48),
            border: egui::Color32::from_rgb(62, 62, 64),
            border_soft: egui::Color32::from_rgb(81, 81, 83),
            warning_fill: egui::Color32::from_rgb(79, 33, 39),
            warning_border: egui::Color32::from_rgb(128, 74, 82),
            warning_text: egui::Color32::from_rgb(244, 216, 218),
            muted_text: egui::Color32::from_rgb(156, 156, 156),
            accent_fill: egui::Color32::from_rgb(14, 99, 156),
            accent_stroke: egui::Color32::from_rgb(17, 119, 187),
            secondary_fill: egui::Color32::from_rgb(45, 45, 48),
            secondary_stroke: egui::Color32::from_rgb(62, 62, 64),
            nav_selected_fill: egui::Color32::from_rgb(15, 118, 110),
            nav_selected_stroke: egui::Color32::from_rgb(17, 94, 89),
            subtab_selected_fill: egui::Color32::from_rgb(55, 55, 61),
            subtab_selected_stroke: egui::Color32::from_rgb(14, 99, 156),
        },
        egui::Theme::Light => ControllerPalette {
            canvas_fill: egui::Color32::from_rgb(241, 244, 247),
            separator: egui::Color32::from_rgb(214, 220, 228),
            card_fill: egui::Color32::from_rgb(255, 255, 255),
            outlined_card_fill: egui::Color32::from_rgb(252, 253, 255),
            border: egui::Color32::from_rgb(214, 220, 228),
            border_soft: egui::Color32::from_rgb(223, 228, 234),
            warning_fill: egui::Color32::from_rgb(254, 242, 242),
            warning_border: egui::Color32::from_rgb(248, 113, 113),
            warning_text: egui::Color32::from_rgb(153, 27, 27),
            muted_text: egui::Color32::from_rgb(79, 93, 112),
            accent_fill: egui::Color32::from_rgb(14, 165, 233),
            accent_stroke: egui::Color32::from_rgb(3, 105, 161),
            secondary_fill: egui::Color32::from_rgb(248, 250, 252),
            secondary_stroke: egui::Color32::from_rgb(203, 213, 225),
            nav_selected_fill: egui::Color32::from_rgb(15, 118, 110),
            nav_selected_stroke: egui::Color32::from_rgb(17, 94, 89),
            subtab_selected_fill: egui::Color32::from_rgb(14, 116, 144),
            subtab_selected_stroke: egui::Color32::from_rgb(12, 74, 110),
        },
    }
}

fn blend_color(left: egui::Color32, right: egui::Color32, amount: f32) -> egui::Color32 {
    let t = amount.clamp(0.0, 1.0);
    let lerp = |a: u8, b: u8| ((a as f32) + ((b as f32) - (a as f32)) * t).round() as u8;
    egui::Color32::from_rgba_premultiplied(
        lerp(left.r(), right.r()),
        lerp(left.g(), right.g()),
        lerp(left.b(), right.b()),
        lerp(left.a(), right.a()),
    )
}

fn is_light_color(color: egui::Color32) -> bool {
    (u16::from(color.r()) + u16::from(color.g()) + u16::from(color.b())) >= (255 * 2) as u16
}

fn themed_badge_colors(
    ui: &egui::Ui,
    fill: egui::Color32,
    color: egui::Color32,
) -> (egui::Color32, egui::Color32) {
    if ui.ctx().theme() != egui::Theme::Dark {
        return (fill, color);
    }
    let palette = controller_palette_for_ui(ui);
    let basis = if is_light_color(color) { fill } else { color };
    let badge_fill = blend_color(palette.outlined_card_fill, basis, 0.28);
    let badge_text = blend_color(egui::Color32::from_rgb(231, 231, 231), basis, 0.32);
    (badge_fill, badge_text)
}

fn themed_metric_colors(
    ui: &egui::Ui,
    accent: egui::Color32,
) -> (egui::Color32, egui::Stroke, egui::Color32) {
    if ui.ctx().theme() != egui::Theme::Dark {
        return (
            egui::Color32::from_rgba_premultiplied(accent.r(), accent.g(), accent.b(), 16),
            egui::Stroke::new(1.0, accent),
            accent,
        );
    }
    let palette = controller_palette_for_ui(ui);
    (
        blend_color(palette.card_fill, accent, 0.12),
        egui::Stroke::new(1.0, blend_color(palette.border_soft, accent, 0.35)),
        blend_color(egui::Color32::from_rgb(232, 232, 232), accent, 0.22),
    )
}

fn controller_palette_for_ctx(ctx: &egui::Context) -> ControllerPalette {
    controller_palette(ctx.theme())
}

fn controller_palette_for_ui(ui: &egui::Ui) -> ControllerPalette {
    controller_palette_for_ctx(ui.ctx())
}

fn controller_visuals(theme: egui::Theme) -> egui::Visuals {
    let palette = controller_palette(theme);
    let mut visuals = theme.default_visuals();
    visuals.panel_fill = palette.canvas_fill;
    visuals.window_fill = palette.card_fill;
    visuals.extreme_bg_color = palette.outlined_card_fill;
    visuals.faint_bg_color = match theme {
        egui::Theme::Dark => egui::Color32::from_rgb(37, 37, 38),
        egui::Theme::Light => egui::Color32::from_rgb(247, 249, 252),
    };
    visuals.widgets.noninteractive.bg_fill = palette.card_fill;
    visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, palette.border);
    visuals.widgets.inactive.bg_fill = palette.card_fill;
    visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, palette.border);
    visuals.widgets.hovered.bg_fill = match theme {
        egui::Theme::Dark => egui::Color32::from_rgb(52, 52, 56),
        egui::Theme::Light => egui::Color32::from_rgb(240, 247, 255),
    };
    visuals.widgets.hovered.bg_stroke = egui::Stroke::new(
        1.0,
        match theme {
            egui::Theme::Dark => palette.border_soft,
            egui::Theme::Light => egui::Color32::from_rgb(37, 99, 235),
        },
    );
    visuals.widgets.active.bg_fill = match theme {
        egui::Theme::Dark => egui::Color32::from_rgb(61, 61, 64),
        egui::Theme::Light => egui::Color32::from_rgb(224, 238, 255),
    };
    visuals.widgets.active.bg_stroke = egui::Stroke::new(
        1.0,
        match theme {
            egui::Theme::Dark => palette.border_soft,
            egui::Theme::Light => egui::Color32::from_rgb(29, 78, 216),
        },
    );
    if theme == egui::Theme::Dark {
        visuals.selection.bg_fill = palette.nav_selected_fill;
        visuals.selection.stroke = egui::Stroke::new(1.0, egui::Color32::WHITE);
        visuals.hyperlink_color = egui::Color32::from_rgb(75, 164, 255);
    }
    visuals
}

fn configure_controller_theme(
    ctx: &egui::Context,
    theme: egui::Theme,
    heading_size: f32,
    body_size: f32,
    button_size: f32,
    small_size: f32,
    monospace_size: f32,
) {
    ctx.style_mut_of(theme, |style| {
        style.visuals = controller_visuals(theme);
        style.spacing.item_spacing = egui::vec2(12.0, 10.0);
        style.spacing.button_padding = egui::vec2(12.0, 7.0);
        style.spacing.indent = 18.0;
        style.spacing.slider_width = 220.0;
        style.spacing.combo_width = 320.0;
        style.spacing.text_edit_width = 280.0;
        style.spacing.interact_size = egui::vec2(44.0, 30.0);
        style.spacing.window_margin = egui::Margin::same(12);
        style.spacing.menu_margin = egui::Margin::same(8);
        style.text_styles.insert(
            egui::TextStyle::Heading,
            egui::FontId::new(heading_size, egui::FontFamily::Proportional),
        );
        style.text_styles.insert(
            egui::TextStyle::Body,
            egui::FontId::new(body_size, egui::FontFamily::Proportional),
        );
        style.text_styles.insert(
            egui::TextStyle::Button,
            egui::FontId::new(button_size, egui::FontFamily::Proportional),
        );
        style.text_styles.insert(
            egui::TextStyle::Small,
            egui::FontId::new(small_size, egui::FontFamily::Proportional),
        );
        style.text_styles.insert(
            egui::TextStyle::Monospace,
            egui::FontId::new(monospace_size, egui::FontFamily::Monospace),
        );
    });
}

pub(super) fn configure_controller_visuals(
    ctx: &egui::Context,
    theme_preference: egui::ThemePreference,
) {
    configure_preferred_fonts(ctx);
    ctx.options_mut(|options| {
        options.tessellation_options = Default::default();
    });

    let ppp = ctx.pixels_per_point().max(0.01);
    let snap = |size: f32| ((size * ppp).round() / ppp).max(1.0);
    configure_controller_theme(
        ctx,
        egui::Theme::Light,
        snap(23.0),
        snap(15.0),
        snap(14.0),
        snap(13.0),
        snap(13.0),
    );
    configure_controller_theme(
        ctx,
        egui::Theme::Dark,
        snap(23.0),
        snap(15.0),
        snap(14.0),
        snap(13.0),
        snap(13.0),
    );
    ctx.set_theme(theme_preference);
}

pub(super) fn render_controller(app: &mut ClusterControllerApp, ctx: &egui::Context) {
    let palette = controller_palette_for_ctx(ctx);
    render_header(app, ctx);
    render_status_banner(app, ctx);
    egui::CentralPanel::default()
        .frame(egui::Frame::default().fill(palette.canvas_fill))
        .show(ctx, |ui| {
            let sidebar_width = 236.0;
            let separator_color = palette.separator;
            ui.horizontal_top(|ui| {
                let left_height = ui.available_height();
                ui.allocate_ui_with_layout(
                    egui::vec2(sidebar_width, left_height),
                    egui::Layout::top_down(egui::Align::Min),
                    |ui| {
                        ui.set_width(sidebar_width);
                        ui.set_min_width(sidebar_width);
                        render_cluster_sidebar(app, ui);
                    },
                );
                let separator_height = ui.available_height().max(left_height);
                let (separator_rect, _) =
                    ui.allocate_exact_size(egui::vec2(1.0, separator_height), egui::Sense::hover());
                ui.painter()
                    .rect_filled(separator_rect, 0.0, separator_color);
                ui.add_space(10.0);
                ui.allocate_ui_with_layout(
                    egui::vec2(ui.available_width(), left_height),
                    egui::Layout::top_down(egui::Align::Min),
                    |ui| {
                        egui::Frame::default()
                            .inner_margin(egui::Margin::symmetric(14, 10))
                            .show(ui, |ui| {
                                egui::ScrollArea::vertical()
                                    .id_salt("cluster-controller-page")
                                    .auto_shrink([false, false])
                                    .show(ui, |ui| match app.selected_page {
                                        ControllerPage::Nodes => render_nodes_page(app, ui),
                                        ControllerPage::Instances => render_instances_page(app, ui),
                                        ControllerPage::Models => render_models_page(app, ui),
                                        ControllerPage::Server => render_server_page(app, ui),
                                        ControllerPage::Settings => render_settings_page(app, ui),
                                        ControllerPage::About => render_about_page(app, ui),
                                    });
                            });
                    },
                );
            });
        });
    render_pairing_request_modal(app, ctx);
}

fn render_header(app: &mut ClusterControllerApp, ctx: &egui::Context) {
    egui::TopBottomPanel::top("controller_header").show(ctx, |ui| {
        ui.add_space(4.0);
        egui::ScrollArea::horizontal()
            .id_salt("controller-header-tabs")
            .auto_shrink([false, true])
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    let refresh_in_progress = app.manual_refresh_in_progress;
                    let refresh_recently_completed = app
                        .last_manual_refresh_completed_at
                        .is_some_and(|finished| finished.elapsed() < Duration::from_secs(2));
                    if header_tab_button(ui, "Nodes", app.selected_page == ControllerPage::Nodes)
                        .clicked()
                    {
                        app.selected_page = ControllerPage::Nodes;
                    }
                    if header_action_button(
                        ui,
                        if refresh_in_progress {
                            "Refreshing..."
                        } else if refresh_recently_completed {
                            "Refreshed"
                        } else {
                            "Refresh"
                        },
                        refresh_in_progress || refresh_recently_completed,
                    )
                    .clicked()
                        && !refresh_in_progress
                    {
                        app.refresh_all_ui();
                    }
                    ui.separator();
                    if header_tab_button(
                        ui,
                        "Instances",
                        app.selected_page == ControllerPage::Instances,
                    )
                    .clicked()
                    {
                        open_instances_loaded_view(app);
                    }
                    if header_tab_button(ui, "Models", app.selected_page == ControllerPage::Models)
                        .clicked()
                    {
                        app.selected_page = ControllerPage::Models;
                    }
                    if header_tab_button(ui, "Server", app.selected_page == ControllerPage::Server)
                        .clicked()
                    {
                        app.selected_page = ControllerPage::Server;
                    }
                    if header_tab_button(
                        ui,
                        "Settings",
                        app.selected_page == ControllerPage::Settings,
                    )
                    .clicked()
                    {
                        app.selected_page = ControllerPage::Settings;
                    }
                    if header_tab_button(ui, "About", app.selected_page == ControllerPage::About)
                        .clicked()
                    {
                        app.selected_page = ControllerPage::About;
                    }
                });
            });
    });
}

fn render_pairing_request_modal(app: &mut ClusterControllerApp, ctx: &egui::Context) {
    let request = active_pairing_request(app).cloned();
    let Some(request) = request else {
        return;
    };

    let backdrop = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::new("pairing-request-backdrop"),
    ));
    backdrop.rect_filled(ctx.content_rect(), 0.0, egui::Color32::from_black_alpha(72));

    egui::Window::new("Node Wants To Pair")
        .id(egui::Id::new("pairing-request-modal"))
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .collapsible(false)
        .resizable(false)
        .movable(false)
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            ui.set_width(420.0);
            ui.label(
                egui::RichText::new(&request.requester_display_name)
                    .strong()
                    .size(18.0),
            );
            ui.add_space(4.0);
            ui.label(format!(
                "{} wants to connect to this node.",
                request.requester_display_name
            ));
            muted_label(
                ui,
                &format!(
                    "{} | {} | {}",
                    request.requester_control_addr,
                    request.requester_os_name,
                    request.requester_arch
                ),
            );
            ui.add_space(8.0);
            outlined_card(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label("Pairing code");
                    state_badge(
                        ui,
                        request.request_code.clone(),
                        egui::Color32::from_rgb(224, 242, 254),
                        egui::Color32::from_rgb(14, 116, 144),
                    );
                });
                muted_label(
                    ui,
                    "Approve to remember this node for future reconnects. Decline to ignore the request.",
                );
            });
            if app.pairing_requests.len() > 1 {
                muted_label(
                    ui,
                    &format!(
                        "{} more pair request(s) are waiting behind this one.",
                        app.pairing_requests.len().saturating_sub(1)
                    ),
                );
            }
            ui.add_space(8.0);
            ui.horizontal_wrapped(|ui| {
                if accent_button(ui, "Pair").clicked() {
                    app.accept_pairing_request(&request.request_id);
                }
                if warning_button(ui, "Decline").clicked() {
                    app.decline_pairing_request(&request.request_id);
                }
            });
        });
}

fn active_pairing_request(app: &ClusterControllerApp) -> Option<&PairingRequestInfo> {
    if let Some(request_id) = &app.pairing_modal_request_id {
        app.pairing_requests
            .iter()
            .find(|request| &request.request_id == request_id)
            .or_else(|| app.pairing_requests.first())
    } else {
        app.pairing_requests.first()
    }
}

fn render_cluster_sidebar(app: &mut ClusterControllerApp, ui: &mut egui::Ui) {
    ui.spacing_mut().item_spacing = egui::vec2(6.0, 6.0);
    ui.spacing_mut().button_padding = egui::vec2(8.0, 4.0);
    egui::ScrollArea::vertical()
        .id_salt("controller-nav-scroll")
        .auto_shrink([false, false])
        .scroll_bar_visibility(egui::containers::scroll_area::ScrollBarVisibility::AlwaysHidden)
        .show(ui, |ui| {
            sidebar_card(ui, Some("Cluster overview"), |ui| {
                render_compact_overview_grid(
                    ui,
                    &[
                        SummaryMetricCard::new(
                            "Nodes",
                            format!("{}", app.telemetry.len().max(app.nodes.len())),
                            "Reachable",
                            egui::Color32::from_rgb(14, 165, 233),
                        ),
                        SummaryMetricCard::new(
                            "Trusted",
                            format!("{}", app.peers.iter().filter(|peer| peer.trusted).count()),
                            "Paired",
                            egui::Color32::from_rgb(34, 197, 94),
                        ),
                        SummaryMetricCard::new(
                            "Loaded",
                            format!("{}", cluster_loaded_instances(app)),
                            "Runtimes",
                            egui::Color32::from_rgb(168, 85, 247),
                        ),
                        SummaryMetricCard::new(
                            "Serving",
                            format!("{}", cluster_serving_instances(app)),
                            "Active",
                            egui::Color32::from_rgb(59, 130, 246),
                        ),
                        SummaryMetricCard::new(
                            "Queued",
                            format!("{}", cluster_queued_requests(app)),
                            "Waiting",
                            egui::Color32::from_rgb(239, 68, 68),
                        ),
                        SummaryMetricCard::new(
                            "Free VRAM",
                            format_mib_from_bytes(total_free_gpu_memory(app)),
                            "GPU free",
                            egui::Color32::from_rgb(249, 115, 22),
                        ),
                    ],
                );
                ui.add_space(6.0);
                render_link_speed_overview_widgets(app, ui);
            });

            sidebar_card(ui, Some("Nodes"), |ui| {
                if app.nodes.is_empty() {
                    ui.label("No nodes discovered yet.");
                    return;
                }
                for node in &app.nodes.clone() {
                    let telemetry = app.telemetry_for_control_addr(&node.control_addr).cloned();
                    let selected = app.selected_control_addr.as_deref().is_some_and(|selected| {
                        lookup_node_for_addr(app, selected)
                            .is_some_and(|current| current.control_addr == node.control_addr)
                    });
                    let palette = controller_palette_for_ui(ui);
                    let mut button = egui::Button::new(
                        egui::RichText::new(&node.node.display_name)
                            .strong()
                            .color(if selected {
                                egui::Color32::WHITE
                            } else {
                                ui.visuals().text_color()
                            }),
                    )
                    .min_size(egui::vec2(ui.available_width(), 30.0))
                    .corner_radius(egui::CornerRadius::same(12));
                    button = if selected {
                        button
                            .fill(palette.nav_selected_fill)
                            .stroke(egui::Stroke::new(1.5, palette.nav_selected_stroke))
                    } else {
                        button
                            .fill(palette.card_fill)
                            .stroke(egui::Stroke::new(1.0, palette.border))
                    };
                    let response = ui.add(button);
                    response.context_menu(|ui| {
                        if ui.button("Refresh telemetry").clicked() {
                            if let Err(err) = app.refresh_telemetry() {
                                app.status = err;
                            } else {
                                app.status =
                                    format!("Telemetry refreshed for {}.", node.node.display_name);
                            }
                            ui.close();
                        }
                        if ui.button("Run link benchmark").clicked() {
                            app.run_cluster_link_benchmarks(true);
                            ui.close();
                        }
                    });
                    if response.clicked() {
                        app.selected_control_addr = Some(node.control_addr.clone());
                        app.selected_instance_id = None;
                        app.instance_creation_open = false;
                        let _ = app.refresh_selected_preview();
                        app.sync_defaults_from_selected_node();
                    }
                    ui.add_space(2.0);
                    wrapped_monospace(ui, &node.control_addr);
                    ui.add_space(1.0);
                    ui.horizontal(|ui| {
                        state_badge(
                            ui,
                            if node.rpc_running { "RPC" } else { "No RPC" },
                            if node.rpc_running {
                                egui::Color32::from_rgb(224, 242, 254)
                            } else {
                                egui::Color32::from_rgb(254, 226, 226)
                            },
                            if node.rpc_running {
                                egui::Color32::from_rgb(14, 116, 144)
                            } else {
                                egui::Color32::from_rgb(153, 27, 27)
                            },
                        );
                        state_badge(
                            ui,
                            if node.public_api_running {
                                "HTTP"
                            } else {
                                "No HTTP"
                            },
                            if node.public_api_running {
                                egui::Color32::from_rgb(220, 252, 231)
                            } else {
                                egui::Color32::from_rgb(254, 249, 195)
                            },
                            if node.public_api_running {
                                egui::Color32::from_rgb(22, 101, 52)
                            } else {
                                egui::Color32::from_rgb(133, 77, 14)
                            },
                        );
                        state_badge(
                            ui,
                            format!("Loaded {}", node_loaded_instances(node, telemetry.as_ref())),
                            egui::Color32::from_rgb(243, 232, 255),
                            egui::Color32::from_rgb(107, 33, 168),
                        );
                    });
                    let visible_devices = filtered_devices_for_node(app, node, telemetry.as_ref());
                    let total_gpu_bytes = visible_devices
                        .iter()
                        .map(|device| device.memory_total)
                        .sum::<u64>();
                    let free_gpu_bytes = visible_devices
                        .iter()
                        .map(|device| device.memory_free)
                        .sum::<u64>();
                    if total_gpu_bytes > 0 {
                        ui.add(
                            egui::ProgressBar::new(memory_ratio(
                                total_gpu_bytes.saturating_sub(free_gpu_bytes),
                                total_gpu_bytes,
                            ))
                            .desired_width(ui.available_width())
                            .text(format!(
                                "GPU free {} / {}",
                                format_mib_from_bytes(free_gpu_bytes),
                                format_mib_from_bytes(total_gpu_bytes)
                            )),
                        );
                    }
                    if !visible_devices.is_empty() {
                        for device in &visible_devices {
                            let used = device.memory_total.saturating_sub(device.memory_free);
                            ui.label(
                                egui::RichText::new(device_display_name_for_ui(app, node, device))
                                    .strong(),
                            );
                            ui.add(
                                egui::ProgressBar::new(memory_ratio(used, device.memory_total))
                                    .desired_width(ui.available_width())
                                    .text(format!(
                                        "{} free / {}",
                                        format_mib(device.memory_free),
                                        format_mib(device.memory_total)
                                    )),
                            );
                        }
                    }
                    if let Some(telemetry) = telemetry.as_ref() {
                        let summary = if app.show_cpu_devices {
                            format!(
                                "Engine RAM {} | system available {}",
                                format_mib_from_bytes(telemetry.process_memory_bytes),
                                format_mib_from_bytes(telemetry.system_memory_available_bytes)
                            )
                        } else {
                            format!(
                                "Engine process RAM {}",
                                format_mib_from_bytes(telemetry.process_memory_bytes)
                            )
                        };
                        muted_label(ui, &summary);
                    }
                    ui.add_space(6.0);
                }
            });
        });
}

fn render_status_banner(app: &mut ClusterControllerApp, ctx: &egui::Context) {
    egui::TopBottomPanel::bottom("controller_status_bar").show(ctx, |ui| {
        ui.horizontal_wrapped(|ui| {
            let status = if app.status.trim().is_empty() {
                "Ready.".to_string()
            } else {
                app.status.clone()
            };
            muted_label(ui, &status);
            if app.model_transfer_in_progress {
                ui.separator();
                ui.spinner();
                muted_label(
                    ui,
                    &format_model_transfer_progress(app.model_transfer_progress.as_ref()),
                );
            }
            if !app.runtime_missing.is_empty() {
                ui.separator();
                state_badge(
                    ui,
                    format!("Runtime issues {}", app.runtime_missing.len()),
                    egui::Color32::from_rgb(254, 226, 226),
                    egui::Color32::from_rgb(153, 27, 27),
                );
                if !app.runtime_install_in_progress
                    && secondary_button(ui, "Install / Repair Runtime").clicked()
                {
                    app.start_runtime_install();
                }
            }
            if app.runtime_install_in_progress {
                ui.spinner();
                muted_label(
                    ui,
                    app.runtime_install_status
                        .as_deref()
                        .unwrap_or("Installing runtime..."),
                );
            }
        });
    });
}

fn format_model_transfer_progress(progress: Option<&crate::ModelTransferProgress>) -> String {
    let Some(progress) = progress else {
        return "Transferring model files...".to_string();
    };
    let file_label = progress.current_file.as_deref().unwrap_or("model file");
    let current_index = (progress.completed_files + 1).min(progress.total_files.max(1));
    format!(
        "Transferring {} from {} to {} | file {}/{} | {} / {} | {}/s",
        file_label,
        progress.source_display_name,
        progress.dest_display_name,
        current_index,
        progress.total_files.max(1),
        format_bytes_compact(progress.transferred_bytes),
        format_bytes_compact(progress.total_bytes),
        format_bytes_compact(progress.bytes_per_second),
    )
}

fn render_overview_page(app: &mut ClusterControllerApp, ui: &mut egui::Ui) {
    ui.label(
        "Automatic discovery, trust prompts, live telemetry, and placement status in one place.",
    );
    ui.add_space(8.0);

    card(ui, Some("Setup checklist"), |ui| {
        let runtime_ready = app.runtime_missing.is_empty();
        let local_connected = !app.nodes.is_empty();
        let trusted_gpu_nodes = app
            .nodes
            .iter()
            .filter(|node| {
                let devices = filtered_devices_for_node(
                    app,
                    node,
                    app.telemetry_for_control_addr(&node.control_addr),
                );
                !devices.is_empty()
            })
            .count();
        let has_models = !app.available_model_packages.is_empty();

        setup_check_item(
            ui,
            runtime_ready,
            "Runtime installed",
            if runtime_ready {
                "Managed runtime is ready on this node."
            } else {
                "Install or repair the managed runtime before loading cluster instances."
            },
        );
        ui.add_space(8.0);
        setup_check_item(
            ui,
            local_connected,
            "Local host connected",
            if local_connected {
                "This controller is attached to the local tray host."
            } else {
                "Connect the local node so the scheduler can plan and serve instances."
            },
        );
        ui.add_space(8.0);
        setup_check_item(
            ui,
            trusted_gpu_nodes > 0,
            "GPU nodes available",
            &format!(
                "{} node{} currently expose eligible GPU targets.",
                trusted_gpu_nodes,
                if trusted_gpu_nodes == 1 { "" } else { "s" }
            ),
        );
        ui.add_space(8.0);
        setup_check_item(
            ui,
            has_models,
            "Available models discovered",
            if has_models {
                "The model store and connected nodes expose model folders for instance creation."
            } else {
                "No available model folders were discovered yet. Open Models to download or import them into AppData."
            },
        );
        ui.add_space(6.0);
        ui.horizontal_wrapped(|ui| {
            if !runtime_ready && accent_button(ui, "Install runtime").clicked() {
                app.start_runtime_install();
            }
            if !local_connected && accent_button(ui, "Connect and look for nodes").clicked() {
                app.connect_local_host_and_start_pair_discovery(180);
            }
            if accent_button(ui, "Open instances").clicked() {
                app.open_instance_creation(true);
            }
            if secondary_button(ui, "Open model store").clicked() {
                app.selected_page = ControllerPage::Models;
            }
        });
    });

    ui.add_space(10.0);

    render_summary_metric_grid(
        ui,
        180.0,
        2,
        &[
            SummaryMetricCard::new(
                "Connected nodes",
                format!("{}", app.telemetry.len().max(app.nodes.len())),
                "Reachable nodes discovered on this network.",
                egui::Color32::from_rgb(14, 165, 233),
            ),
            SummaryMetricCard::new(
                "Model folders",
                format!("{}", app.available_model_packages.len()),
                "Available model folders across this node and connected peers.",
                egui::Color32::from_rgb(34, 197, 94),
            ),
        ],
    );

    ui.add_space(12.0);
    card(ui, Some("Node health"), |ui| {
        if app.telemetry.is_empty() {
            ui.label("No live telemetry yet.");
            return;
        }
        for snapshot in &app.telemetry.clone() {
            let node = app
                .nodes
                .iter()
                .find(|node| {
                    node.control_addr == snapshot.control_addr
                        || node.advertised_control_addr.as_deref()
                            == snapshot.advertised_control_addr.as_deref()
                })
                .cloned();
            outlined_card(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label(
                        egui::RichText::new(&snapshot.node.display_name)
                            .strong()
                            .size(17.0),
                    );
                    muted_label(ui, &snapshot.node.os_name);
                    muted_label(ui, &snapshot.node.arch);
                    if app.show_cpu_devices {
                        state_badge(
                            ui,
                            format!("Host CPU {:.1}%", snapshot.process_cpu_percent),
                            egui::Color32::from_rgb(254, 249, 195),
                            egui::Color32::from_rgb(133, 77, 14),
                        );
                    }
                    state_badge(
                        ui,
                        format!(
                            "Serving {}",
                            snapshot
                                .instances
                                .iter()
                                .filter(|item| item.active_request_count > 0 || item.state == 3)
                                .count()
                        ),
                        egui::Color32::from_rgb(224, 242, 254),
                        egui::Color32::from_rgb(14, 116, 144),
                    );
                    let queued_requests = snapshot
                        .instances
                        .iter()
                        .map(|item| item.queued_request_count.max(0))
                        .sum::<i32>();
                    if queued_requests > 0 {
                        state_badge(
                            ui,
                            format!("Queued {}", queued_requests),
                            egui::Color32::from_rgb(254, 242, 242),
                            egui::Color32::from_rgb(185, 28, 28),
                        );
                    }
                });
                ui.label(format!(
                    "Control {}{}{}",
                    snapshot.control_addr,
                    node.as_ref()
                        .and_then(|value| value.advertised_public_api_addr.as_ref())
                        .map(|addr| format!(" | HTTP {addr}"))
                        .unwrap_or_default(),
                    node.as_ref()
                        .and_then(|value| value.advertised_rpc_endpoint.as_ref())
                        .map(|addr| format!(" | RPC {addr}"))
                        .unwrap_or_default()
                ));
                let visible_devices = node
                    .as_ref()
                    .map(|value| filtered_devices_for_node(app, value, Some(snapshot)))
                    .unwrap_or_default();
                let total_gpu_bytes = visible_devices
                    .iter()
                    .map(|device| device.memory_total)
                    .sum::<u64>();
                let free_gpu_bytes = visible_devices
                    .iter()
                    .map(|device| device.memory_free)
                    .sum::<u64>();
                if total_gpu_bytes > 0 {
                    ui.add(
                        egui::ProgressBar::new(memory_ratio(
                            total_gpu_bytes.saturating_sub(free_gpu_bytes),
                            total_gpu_bytes,
                        ))
                        .desired_width(ui.available_width())
                        .text(format!(
                            "GPU free {} of {}",
                            format_mib_from_bytes(free_gpu_bytes),
                            format_mib_from_bytes(total_gpu_bytes)
                        )),
                    );
                }
                let process_summary = if app.show_cpu_devices {
                    format!(
                        "Engine RAM {} | host available RAM {}",
                        format_mib_from_bytes(snapshot.process_memory_bytes),
                        format_mib_from_bytes(snapshot.system_memory_available_bytes)
                    )
                } else {
                    format!(
                        "Engine RAM {}",
                        format_mib_from_bytes(snapshot.process_memory_bytes)
                    )
                };
                muted_label(ui, &process_summary);
                ui.add_space(4.0);
                for device in &visible_devices {
                    let used = device.memory_total.saturating_sub(device.memory_free);
                    ui.add(
                        egui::ProgressBar::new(memory_ratio(used, device.memory_total))
                            .desired_width(ui.available_width())
                            .text(format!(
                                "{} [{}] {} free of {}",
                                node.as_ref()
                                    .map(|value| device_display_name_for_ui(app, value, device))
                                    .unwrap_or_else(|| device.name.clone()),
                                device.backend,
                                format_mib(device.memory_free),
                                format_mib(device.memory_total)
                            )),
                    );
                    if let Some(node) = node.as_ref() {
                        render_device_instance_summary(app, ui, snapshot, node, device);
                    }
                }
                let loaded_instances = snapshot
                    .instances
                    .iter()
                    .filter(|instance| instance.state != 0)
                    .cloned()
                    .collect::<Vec<_>>();
                if !loaded_instances.is_empty() {
                    ui.add_space(6.0);
                    ui.horizontal_wrapped(|ui| {
                        for instance in loaded_instances {
                            summary_pill(
                                ui,
                                instance.name,
                                instance_chip_color(instance.instance_id),
                                egui::Color32::WHITE,
                            );
                        }
                    });
                }
                if !snapshot.link_metrics.is_empty() {
                    ui.add_space(6.0);
                    ui.label(egui::RichText::new("Inter-node links").strong());
                    for link in &snapshot.link_metrics {
                        ui.label(format_link_metrics(link));
                    }
                }
                ui.horizontal_wrapped(|ui| {
                    if accent_button(ui, "Inspect telemetry").clicked() {
                        app.selected_control_addr = Some(snapshot.control_addr.clone());
                        app.selected_page = ControllerPage::Instances;
                        app.instance_creation_open = false;
                    }
                    if secondary_button(ui, "View instances").clicked() {
                        app.selected_control_addr = Some(snapshot.control_addr.clone());
                        open_instances_loaded_view(app);
                    }
                });
            });
        }
    });

    if let Some(plan) = &app.last_plan {
        ui.add_space(10.0);
        card(ui, Some("Last placement plan"), |ui| {
            ui.label(format!(
                "{} on {} via {}",
                placement_strategy_label(plan.strategy),
                plan.owner_display_name,
                plan.execution_group_id
            ));
            ui.label(format!(
                "Estimated required {} | free {} | reusable {} | ready {} | eviction {}",
                format_mib_from_bytes(plan.estimated_required_bytes),
                format_mib_from_bytes(plan.estimated_group_free_bytes),
                plan.reusable_instance_id
                    .map(|value: i64| value.to_string())
                    .unwrap_or_else(|| "<none>".to_string()),
                yes_no(plan.ready_now),
                yes_no(plan.requires_eviction),
            ));
            if !plan.rpc_servers.is_empty() {
                ui.label(format!("Remote workers: {}", plan.rpc_servers));
            }
        });
    }
}

fn render_models_page(app: &mut ClusterControllerApp, ui: &mut egui::Ui) {
    ui.label("Download model files from Hugging Face, import local files, or use the supported audio shortcuts. Available models merge this node with connected peers and show where each selected file exists right now.");
    ui.add_space(8.0);

    card(ui, Some("Available models"), |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label("Models directory");
            wrapped_monospace(ui, &app.local_models_dir().display().to_string());
            if secondary_button(ui, "Refresh models").clicked() {
                app.refresh_all_ui();
            }
            if secondary_button(ui, "Open folder").clicked() {
                app.open_local_models_folder();
            }
        });
        ui.label(format!(
            "{} available folder(s) across this node and connected peers.",
            app.available_model_packages.len()
        ));
    });

    ui.add_space(8.0);
    ui.horizontal_wrapped(|ui| {
        if subtab_button(
            ui,
            "Available models",
            app.model_store_mode == ModelStoreMode::LocalInstalled,
        )
        .clicked()
        {
            app.model_store_mode = ModelStoreMode::LocalInstalled;
        }
        if subtab_button(
            ui,
            "Download from Hugging Face",
            app.model_store_mode == ModelStoreMode::RepoBrowser,
        )
        .clicked()
        {
            app.model_store_mode = ModelStoreMode::RepoBrowser;
        }
        if subtab_button(
            ui,
            "Import local files",
            app.model_store_mode == ModelStoreMode::ImportLocal,
        )
        .clicked()
        {
            app.model_store_mode = ModelStoreMode::ImportLocal;
        }
        if subtab_button(
            ui,
            "Supported audio models",
            app.model_store_mode == ModelStoreMode::SupportedAudio,
        )
        .clicked()
        {
            app.model_store_mode = ModelStoreMode::SupportedAudio;
        }
    });

    ui.add_space(8.0);
    match app.model_store_mode {
        ModelStoreMode::LocalInstalled => render_local_installed_models_page(app, ui),
        ModelStoreMode::RepoBrowser => render_repo_browser_page(app, ui),
        ModelStoreMode::ImportLocal => render_import_model_page(app, ui),
        ModelStoreMode::SupportedAudio => render_supported_audio_model_page(app, ui),
    }
}

fn render_local_installed_models_page(app: &mut ClusterControllerApp, ui: &mut egui::Ui) {
    card(ui, Some("Available models"), |ui| {
        ui.label(
            "These are the model folders currently available on this node and connected peers. Select one to inspect its files, current node availability, and any local README.",
        );
        ui.add_space(8.0);
        if app.available_model_packages.is_empty() {
            ui.label("No available model folders yet.");
            return;
        }
        render_local_model_package_list(app, ui);
    });
    ui.add_space(12.0);
    card(ui, Some("Selected package"), |ui| {
        let Some(package) = app.selected_model_package().cloned() else {
            ui.label("Choose an available model folder to inspect it here.");
            return;
        };
        let package_details = app.selected_model_package_detail().cloned();
        let local_package = app.selected_local_model_package().cloned();
        ui.label(
            egui::RichText::new(&package.display_name)
                .strong()
                .size(18.0),
        );
        ui.label(format!("Folder: {}", package.folder_name));
        if let Some(details) = &package_details {
            render_package_node_summary(ui, &details.available_on);
        }
        if let Some(local_package) = &local_package {
            wrapped_monospace(ui, &local_package.path.display().to_string());
        } else if let Some(details) = &package_details {
            if let Some(location) = details.available_on.first() {
                wrapped_monospace(ui, &location.package_path);
            }
        }
        ui.horizontal_wrapped(|ui| {
            if secondary_button_enabled(ui, "Open package folder", local_package.is_some())
                .clicked()
            {
                app.open_selected_model_package_folder();
            }
            if accent_button(ui, "Use in instance setup").clicked() {
                app.open_instance_creation(true);
            }
        });
        ui.add_space(6.0);
        outlined_card(ui, |ui| {
            ui.label(egui::RichText::new("Files").strong());
            render_package_artifact_inventory(app, ui, package_details.as_ref(), &package);
        });
        if let Some(readme) = local_package.as_ref().and_then(load_local_package_readme) {
            ui.add_space(8.0);
            outlined_card(ui, |ui| {
                ui.label(egui::RichText::new("README").strong());
                render_readme_preview(app, ui, "local-installed-package-readme", &readme, 0.0);
            });
        }
    });
}

fn render_local_model_package_list(app: &mut ClusterControllerApp, ui: &mut egui::Ui) {
    for package in &app.available_model_packages.clone() {
        let selected =
            app.selected_model_package_folder.as_deref() == Some(package.folder_name.as_str());
        let details = app
            .available_model_package_details
            .get(&package.folder_name)
            .cloned();
        outlined_card(ui, |ui| {
            if ui
                .selectable_label(
                    selected,
                    egui::RichText::new(&package.display_name).strong(),
                )
                .clicked()
            {
                app.selected_model_package_folder = Some(package.folder_name.clone());
                app.sync_selected_model_package();
            }
            muted_label(ui, &format!("Folder: {}", package.folder_name));
            if let Some(details) = &details {
                render_package_node_summary(ui, &details.available_on);
            }
            ui.horizontal_wrapped(|ui| {
                state_badge(
                    ui,
                    format!("{} model file(s)", package.model_files.len()),
                    egui::Color32::from_rgb(224, 242, 254),
                    egui::Color32::from_rgb(14, 116, 144),
                );
                if !package.mmproj_files.is_empty() {
                    state_badge(
                        ui,
                        format!("{} mmproj", package.mmproj_files.len()),
                        egui::Color32::from_rgb(254, 249, 195),
                        egui::Color32::from_rgb(161, 98, 7),
                    );
                }
                if let Some(repo_id) = &package.guessed_repo_id {
                    muted_label(ui, repo_id);
                }
            });
            render_package_artifact_inventory(app, ui, details.as_ref(), package);
        });
    }
}

fn render_package_node_summary(
    ui: &mut egui::Ui,
    availability: &[crate::protocol::ModelPackageNodeAvailability],
) {
    if availability.is_empty() {
        muted_label(
            ui,
            "No connected nodes are reporting this folder right now.",
        );
        return;
    }
    ui.horizontal_wrapped(|ui| {
        ui.label("Nodes");
        for node in availability {
            state_badge(
                ui,
                &node.display_name,
                egui::Color32::from_rgb(224, 242, 254),
                egui::Color32::from_rgb(14, 116, 144),
            );
        }
    });
}

fn render_package_artifact_inventory(
    app: &mut ClusterControllerApp,
    ui: &mut egui::Ui,
    details: Option<&ClusterModelPackageInfo>,
    package: &crate::model_store::ModelPackage,
) {
    if package.model_files.is_empty() && package.mmproj_files.is_empty() {
        muted_label(ui, "No model files were discovered in this folder.");
        return;
    }

    for file in &package.model_files {
        let availability = details.and_then(|details| {
            details
                .model_file_availability
                .iter()
                .find(|entry| entry.artifact.relative_path == file.relative_path)
        });
        render_package_artifact_row(
            app,
            ui,
            &package.folder_name,
            &file.relative_path,
            &file.relative_path,
            file.size_bytes,
            availability,
        );
    }
    for file in &package.mmproj_files {
        let availability = details.and_then(|details| {
            details
                .mmproj_file_availability
                .iter()
                .find(|entry| entry.artifact.relative_path == file.relative_path)
        });
        render_package_artifact_row(
            app,
            ui,
            &package.folder_name,
            &format!("mmproj: {}", file.relative_path),
            &file.relative_path,
            file.size_bytes,
            availability,
        );
    }
}

fn render_package_artifact_row(
    app: &mut ClusterControllerApp,
    ui: &mut egui::Ui,
    package_folder: &str,
    label: &str,
    relative_path: &str,
    size_bytes: u64,
    availability: Option<&ClusterModelArtifactInfo>,
) {
    ui.horizontal_wrapped(|ui| {
        wrapped_monospace(ui, label);
        muted_label(ui, &format_bytes_compact(size_bytes));
        if let Some(availability) = availability {
            render_package_artifact_actions(app, ui, package_folder, relative_path, availability);
        }
    });
    let nodes = availability
        .map(|availability| {
            availability
                .available_on
                .iter()
                .map(|node| node.display_name.clone())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if nodes.is_empty() {
        muted_label(ui, "Connected nodes reporting this file: none");
    } else {
        muted_label(
            ui,
            &format!("Connected nodes reporting this file: {}", nodes.join(", ")),
        );
    }
}

fn render_package_artifact_actions(
    app: &mut ClusterControllerApp,
    ui: &mut egui::Ui,
    package_folder: &str,
    relative_path: &str,
    availability: &ClusterModelArtifactInfo,
) {
    let local_control_addr = app.host.control_addr().to_string();
    let has_local = availability
        .available_on
        .iter()
        .any(|node| node.control_addr == local_control_addr);
    let missing_remote_nodes = app
        .nodes
        .iter()
        .filter(|node| node.control_addr != local_control_addr)
        .filter(|node| {
            !availability
                .available_on
                .iter()
                .any(|location| location.control_addr == node.control_addr)
        })
        .map(|node| (node.control_addr.clone(), node.node.display_name.clone()))
        .collect::<Vec<_>>();

    if !has_local
        && availability
            .available_on
            .iter()
            .any(|node| node.control_addr != local_control_addr)
    {
        if secondary_button_enabled(
            ui,
            "Retrieve to this machine",
            !app.model_transfer_in_progress,
        )
        .clicked()
        {
            app.start_single_artifact_transfer_to_node(
                package_folder,
                relative_path,
                &local_control_addr,
            );
        }
    }

    if has_local && !missing_remote_nodes.is_empty() {
        if missing_remote_nodes.len() == 1 {
            let (dest_control_addr, dest_display_name) = &missing_remote_nodes[0];
            let upload_label = format!("Upload to {}", dest_display_name);
            if secondary_button_enabled(ui, &upload_label, !app.model_transfer_in_progress)
                .clicked()
            {
                app.start_single_artifact_transfer_to_node(
                    package_folder,
                    relative_path,
                    dest_control_addr,
                );
            }
        } else {
            ui.menu_button("Upload to node", |ui| {
                for (dest_control_addr, dest_display_name) in &missing_remote_nodes {
                    if ui
                        .button(format!("Upload to {}", dest_display_name))
                        .clicked()
                    {
                        app.start_single_artifact_transfer_to_node(
                            package_folder,
                            relative_path,
                            dest_control_addr,
                        );
                        ui.close();
                    }
                }
            });
        }
    }
}

fn render_repo_browser_page(app: &mut ClusterControllerApp, ui: &mut egui::Ui) {
    card(ui, Some("Download from Hugging Face"), |ui| {
        ui.label("Paste a Hugging Face repo, load the file list, pick what you want, then download directly into the AppData models directory.");
        ui.horizontal_wrapped(|ui| {
            ui.label("Repo");
            ui.add_enabled(
                app.model_store_busy == ModelStoreBusyState::Idle,
                egui::TextEdit::singleline(&mut app.model_store_repo_input)
                    .hint_text("owner/name or https://huggingface.co/owner/name")
                    .desired_width(adaptive_field_width(ui, 0.72, 260.0, 440.0)),
            );
            if accent_button(ui, "Load repo").clicked()
                && app.model_store_busy == ModelStoreBusyState::Idle
            {
                app.start_current_repo_load();
            }
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("Folder name");
            ui.add_enabled(
                app.model_store_busy == ModelStoreBusyState::Idle,
                egui::TextEdit::singleline(&mut app.model_store_repo_folder_name)
                    .hint_text("owner__repo")
                    .desired_width(adaptive_field_width(ui, 0.48, 220.0, 320.0)),
            );
            let can_download = app
                .model_store_repo_preview
                .as_ref()
                .map(|preview| preview.files.iter().any(|file| file.selected))
                .unwrap_or(false)
                && app.model_store_busy == ModelStoreBusyState::Idle;
            if secondary_button_enabled(ui, "Download selected", can_download).clicked() {
                app.start_repo_download();
            }
        });
        if let Some(error) = &app.model_store_error {
            warning_card(ui, "Model store error", error);
        }
        if app.model_store_busy == ModelStoreBusyState::Downloading {
            let progress = if app.model_store_progress.total_bytes == 0 {
                0.0
            } else {
                app.model_store_progress.downloaded_bytes as f32
                    / app.model_store_progress.total_bytes as f32
            };
            ui.add(
                egui::ProgressBar::new(progress.clamp(0.0, 1.0))
                    .desired_width(ui.available_width())
                    .text(download_progress_summary(&app.model_store_progress)),
            );
            if let Some(current) = &app.model_store_progress.current_file {
                muted_label(ui, &format!("Current file: {current}"));
            }
        }
        ui.add_space(8.0);
        if let Some(preview) = &mut app.model_store_repo_preview {
            ui.horizontal_wrapped(|ui| {
                ui.label("Repo URL");
                ui.hyperlink_to(&preview.repo_url, &preview.repo_url);
            });
            ui.label(format!("Revision: {}", preview.revision));
            ui.add_space(6.0);
            ui.horizontal_wrapped(|ui| {
                let mut all_selected = preview.files.iter().all(|file| file.selected);
                if ui.checkbox(&mut all_selected, "Select all").changed() {
                    for file in &mut preview.files {
                        file.selected = all_selected;
                    }
                }
                ui.label(format!("{} file(s) listed", preview.files.len()));
            });
            for file in &mut preview.files {
                ui.horizontal_wrapped(|ui| {
                    ui.checkbox(&mut file.selected, "");
                    wrapped_monospace(ui, &file.path);
                    if let Some(size) = file.size {
                        muted_label(ui, &format_bytes_compact(size));
                    }
                });
            }
        } else {
            ui.label("Load a repo to inspect its files.");
        }
    });
    ui.add_space(12.0);
    card(ui, Some("Repo README"), |ui| {
        if let Some(preview) = &app.model_store_repo_preview {
            let readme_markdown = preview.readme_markdown.clone();
            ui.label(egui::RichText::new(&preview.repo_id).strong());
            ui.hyperlink_to(
                "Open README on Hugging Face",
                format!("{}/blob/{}/README.md", preview.repo_url, preview.revision),
            );
            if let Some(readme) = readme_markdown.as_deref() {
                render_readme_preview(app, ui, "repo-readme-preview", readme, 0.0);
            } else {
                ui.label("This repo did not expose a README.md in the file listing.");
            }
        } else {
            ui.label("Repo README preview appears here after loading a repo.");
        }
    });
}

fn render_import_model_page(app: &mut ClusterControllerApp, ui: &mut egui::Ui) {
    card(ui, Some("Import local files"), |ui| {
        ui.label(
            "Pick model files from disk, choose the folder name to create in AppData, and import them directly into the local model store.",
        );
        ui.add_space(8.0);
        ui.horizontal_wrapped(|ui| {
            ui.label("Model folder name");
            ui.add_enabled(
                app.model_store_busy == ModelStoreBusyState::Idle,
                egui::TextEdit::singleline(&mut app.model_store_import_name)
                    .hint_text("folder name in AppData")
                    .desired_width(adaptive_field_width(ui, 0.54, 220.0, 320.0)),
            );
        });
        ui.horizontal_wrapped(|ui| {
            if secondary_button(ui, "Pick files").clicked()
                && app.model_store_busy == ModelStoreBusyState::Idle
            {
                app.pick_import_files();
            }
            if secondary_button_enabled(
                ui,
                "Import selected files",
                app.model_store_busy == ModelStoreBusyState::Idle
                    && !app.model_store_import_files.is_empty(),
            )
            .clicked()
            {
                app.start_local_import();
            }
        });
        ui.add_space(8.0);
        if app.model_store_import_files.is_empty() {
            muted_label(ui, "No local files selected yet.");
        } else {
            egui::ScrollArea::vertical()
                .id_salt("local-import-files")
                .max_height(360.0)
                .show(ui, |ui| {
                    for file in &app.model_store_import_files {
                        wrapped_monospace(ui, &file.display().to_string());
                    }
                });
        }
        if let Some(error) = &app.model_store_error {
            ui.add_space(8.0);
            warning_card(ui, "Model store error", error);
        }
    });
}

fn render_supported_audio_model_page(app: &mut ClusterControllerApp, ui: &mut egui::Ui) {
    card(ui, Some("Supported audio models"), |ui| {
        ui.label("These are the native bridge model families ENGINE supports directly for transcription, realtime audio, and diarization.");
        ui.add_space(8.0);
        for repo in supported_audio_repos() {
            let selected = app.selected_supported_audio_repo == repo;
            outlined_card(ui, |ui| {
                if ui
                    .selectable_label(selected, egui::RichText::new(repo.title()).strong())
                    .clicked()
                {
                    app.load_supported_audio_repo(repo);
                }
                muted_label(ui, repo.description());
                ui.hyperlink_to(repo.repo_id(), repo.repo_url());
            });
        }
    });
    ui.add_space(12.0);
    card(ui, Some("Repo details"), |ui| {
        let selected = app.selected_supported_audio_repo;
        ui.label(egui::RichText::new(selected.title()).strong().size(18.0));
        muted_label(ui, selected.description());
        ui.hyperlink_to(selected.repo_id(), selected.repo_url());
        ui.add_space(8.0);
        if secondary_button(ui, "Reload repo details").clicked()
            && app.model_store_busy == ModelStoreBusyState::Idle
        {
            app.load_supported_audio_repo(selected);
        }
        if app.model_store_repo_preview.is_none()
            && app.model_store_busy == ModelStoreBusyState::Idle
        {
            app.load_supported_audio_repo(selected);
        }
        let mut trigger_download = false;
        let mut readme_markdown = None;
        if let Some(preview) = &mut app.model_store_repo_preview {
            ui.horizontal_wrapped(|ui| {
                ui.label("Folder name");
                ui.text_edit_singleline(&mut app.model_store_repo_folder_name);
            });
            ui.hyperlink_to(
                "Open README on Hugging Face",
                format!("{}/blob/{}/README.md", preview.repo_url, preview.revision),
            );
            if secondary_button_enabled(
                ui,
                "Download selected",
                app.model_store_busy == ModelStoreBusyState::Idle
                    && preview.files.iter().any(|file| file.selected),
            )
            .clicked()
            {
                trigger_download = true;
            }
            ui.add_space(6.0);
            for file in &mut preview.files {
                ui.horizontal_wrapped(|ui| {
                    ui.checkbox(&mut file.selected, "");
                    wrapped_monospace(ui, &file.path);
                    if let Some(size) = file.size {
                        muted_label(ui, &format_bytes_compact(size));
                    }
                });
            }
            ui.add_space(8.0);
            readme_markdown = preview.readme_markdown.clone();
        } else if let Some(error) = &app.model_store_error {
            warning_card(ui, "Model store error", error);
        } else {
            ui.label("Loading repo details...");
        }
        if let Some(readme) = readme_markdown.as_deref() {
            render_readme_preview(app, ui, "supported-audio-readme", readme, 0.0);
        } else if app.model_store_repo_preview.is_some() {
            ui.label("README preview unavailable for this repo.");
        }
        if trigger_download {
            app.start_repo_download();
        }
    });
}

fn render_model_package_inventory(app: &mut ClusterControllerApp, ui: &mut egui::Ui) {
    ui.label(egui::RichText::new("Available model folders").strong());
    if app.available_model_packages.is_empty() {
        ui.label("No available model folders yet.");
        return;
    }
    egui::ScrollArea::vertical()
        .id_salt("local-model-package-list")
        .max_height(220.0)
        .show(ui, |ui| {
            for package in &app.available_model_packages {
                outlined_card(ui, |ui| {
                    ui.label(egui::RichText::new(&package.display_name).strong());
                    muted_label(ui, &format!("Folder: {}", package.folder_name));
                    muted_label(
                        ui,
                        &format!(
                            "{} model file(s) | {} mmproj file(s)",
                            package.model_files.len(),
                            package.mmproj_files.len()
                        ),
                    );
                    if let Some(repo_id) = &package.guessed_repo_id {
                        muted_label(ui, &format!("Guessed repo: {repo_id}"));
                    }
                    if let Some(readme) = load_local_package_readme(package) {
                        let snippet = readme.lines().take(3).collect::<Vec<_>>().join(" ");
                        if !snippet.trim().is_empty() {
                            muted_label(ui, &snippet);
                        }
                    }
                });
            }
        });
}

fn instance_runtime_kind_options() -> [(&'static str, &'static str); 7] {
    [
        ("text", "Text"),
        ("vision", "Vision"),
        ("embeddings", "Embeddings"),
        ("rerank", "Rerank"),
        ("whisper", "Whisper"),
        ("realtime-audio", "Realtime audio"),
        ("diarization", "Diarization"),
    ]
}

fn render_instances_page(app: &mut ClusterControllerApp, ui: &mut egui::Ui) {
    ui.label("Create new runtimes from available model folders across this node and connected peers, then inspect what is already loaded across the cluster.");
    ui.add_space(8.0);

    ui.horizontal_wrapped(|ui| {
        if app.instance_creation_open {
            if secondary_button(ui, "Show loaded runtimes").clicked() {
                app.instance_creation_open = false;
            }
            if accent_button(ui, "Create new instance").clicked() {
                app.open_instance_creation(true);
            }
        } else {
            if accent_button(ui, "Show loaded runtimes").clicked() {
                app.instance_creation_open = false;
            }
            if secondary_button(ui, "Create new instance").clicked() {
                app.open_instance_creation(true);
            }
        }
    });

    if !app.instance_presets.is_empty() {
        ui.add_space(8.0);
        card(ui, Some("Quick start preset"), |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label("Preset");
                let previous = app.selected_instance_preset_name.clone();
                egui::ComboBox::from_id_salt("instance-quick-preset")
                    .selected_text(
                        app.selected_instance_preset_name
                            .clone()
                            .unwrap_or_else(|| "Choose a preset".to_string()),
                    )
                    .width(adaptive_combo_width(ui, 0.42, 240.0, 360.0))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut app.selected_instance_preset_name,
                            None,
                            "Choose a preset",
                        );
                        for preset in &app.instance_presets {
                            ui.selectable_value(
                                &mut app.selected_instance_preset_name,
                                Some(preset.name.clone()),
                                &preset.name,
                            );
                        }
                    });
                if previous != app.selected_instance_preset_name {
                    if let Some(selected) = app.selected_instance_preset_name.clone() {
                        app.apply_instance_preset_by_name(&selected);
                    }
                }
                if secondary_button_enabled(
                    ui,
                    "Load preset now",
                    app.selected_instance_preset().is_some(),
                )
                .clicked()
                {
                    app.schedule_instance_cluster();
                    open_instances_loaded_view(app);
                }
                if warning_button(ui, "Delete preset").clicked()
                    && app.selected_instance_preset().is_some()
                {
                    app.delete_selected_instance_preset();
                }
            });
            muted_label(
                ui,
                "Presets remember model file choices, placement target, runtime defaults, and max predict so you can relaunch without reconfiguring everything.",
            );
        });
    }

    ui.add_space(8.0);
    if app.instance_creation_open {
        card(ui, Some("Create new instance"), |ui| {
            ui.label(
                "Pick a model type, choose an available model folder, then choose which files to load. Placement, split settings, and node selection stay below.",
            );
            ui.add_space(8.0);

            ui.horizontal_wrapped(|ui| {
                ui.label("Model type");
                let previous_kind = app.instance_model_kind.clone();
                egui::ComboBox::from_id_salt("instance-runtime-kind")
                    .selected_text(runtime_kind_label(&app.instance_model_kind))
                    .width(adaptive_combo_width(ui, 0.34, 180.0, 260.0))
                    .show_ui(ui, |ui| {
                        for (value, label) in instance_runtime_kind_options() {
                            ui.selectable_value(
                                &mut app.instance_model_kind,
                                value.to_string(),
                                label,
                            );
                        }
                    });
                if previous_kind != app.instance_model_kind {
                    app.sync_load_on_demand_grace_for_kind_change(&previous_kind);
                    app.sync_selected_model_package();
                }
            });

            if app.available_model_packages.is_empty() {
                warning_card(
                    ui,
                    "No available model folders",
                    "Download or import model files first, or connect to another node that already has them. Available model folders merge the AppData store on this node with connected peers.",
                );
                ui.add_space(8.0);
                if accent_button(ui, "Open model store").clicked() {
                    app.selected_page = ControllerPage::Models;
                }
                return;
            }

            ui.horizontal_wrapped(|ui| {
                ui.label("Model folder");
                let previous_folder = app.selected_model_package_folder.clone();
                let selected_text = app
                    .selected_model_package()
                    .map(|package| package.display_name.clone())
                    .unwrap_or_else(|| "Choose an available model folder".to_string());
                egui::ComboBox::from_id_salt("instance-model-folder")
                    .selected_text(selected_text)
                    .width(adaptive_combo_width(ui, 0.62, 260.0, 520.0))
                    .show_ui(ui, |ui| {
                        for package in app.available_model_packages.clone() {
                            let node_count = app
                                .available_model_package_details
                                .get(&package.folder_name)
                                .map(|details| details.available_on.len())
                                .unwrap_or(0);
                            let label = format!(
                                "{} ({} model file(s), {} mmproj file(s), {} node{})",
                                package.display_name,
                                package.model_files.len(),
                                package.mmproj_files.len(),
                                node_count,
                                if node_count == 1 { "" } else { "s" }
                            );
                            ui.selectable_value(
                                &mut app.selected_model_package_folder,
                                Some(package.folder_name.clone()),
                                label,
                            );
                        }
                    });
                if previous_folder != app.selected_model_package_folder {
                    app.selected_model_file_path = None;
                    app.selected_mmproj_file_path = None;
                    app.selected_diarization_file_path = None;
                    app.sync_selected_model_package();
                }
            });

            let selected_package = app.selected_model_package().cloned();
            let selected_package_details = app.selected_model_package_detail().cloned();
            if let Some(package) = selected_package {
                if let Some(details) = &selected_package_details {
                    render_package_node_summary(ui, &details.available_on);
                }
                ui.horizontal_wrapped(|ui| {
                    ui.label("Primary model file");
                    let previous_file = app.selected_model_file_path.clone();
                    let selected_text = app
                        .selected_model_file_path
                        .clone()
                        .unwrap_or_else(|| "Choose a model file".to_string());
                    egui::ComboBox::from_id_salt("instance-model-file")
                        .selected_text(selected_text)
                        .width(adaptive_combo_width(ui, 0.72, 280.0, 620.0))
                        .show_ui(ui, |ui| {
                            for file in &package.model_files {
                                ui.selectable_value(
                                    &mut app.selected_model_file_path,
                                    Some(file.relative_path.clone()),
                                    format!(
                                        "{} ({})",
                                        file.relative_path,
                                        format_bytes_compact(file.size_bytes)
                                    ),
                                );
                            }
                        });
                    if previous_file != app.selected_model_file_path {
                        app.sync_selected_model_package();
                    }
                });
                if let (Some(details), Some(selected_file)) = (
                    selected_package_details.as_ref(),
                    app.selected_model_file_path.as_ref(),
                ) {
                    if let Some(availability) = details
                        .model_file_availability
                        .iter()
                        .find(|entry| &entry.artifact.relative_path == selected_file)
                    {
                        muted_label(
                            ui,
                            &format!(
                                "Primary file is currently on: {}",
                                availability
                                    .available_on
                                    .iter()
                                    .map(|node| node.display_name.clone())
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            ),
                        );
                    }
                }

                if app.instance_model_kind == "vision" {
                    if package.mmproj_files.is_empty() {
                        warning_card(
                            ui,
                            "Vision needs an MMProj file",
                            "This folder does not contain an mmproj file. Pick a different folder or switch the model type back to Text.",
                        );
                    } else {
                        ui.horizontal_wrapped(|ui| {
                            ui.label("MMProj file");
                            let previous_mmproj = app.selected_mmproj_file_path.clone();
                            let selected_text = app
                                .selected_mmproj_file_path
                                .clone()
                                .unwrap_or_else(|| "Choose an mmproj file".to_string());
                            egui::ComboBox::from_id_salt("instance-mmproj-file")
                                .selected_text(selected_text)
                                .width(adaptive_combo_width(ui, 0.72, 280.0, 620.0))
                                .show_ui(ui, |ui| {
                                    for file in &package.mmproj_files {
                                        ui.selectable_value(
                                            &mut app.selected_mmproj_file_path,
                                            Some(file.relative_path.clone()),
                                            format!(
                                                "{} ({})",
                                                file.relative_path,
                                                format_bytes_compact(file.size_bytes)
                                            ),
                                        );
                                    }
                                });
                            if previous_mmproj != app.selected_mmproj_file_path {
                                app.sync_selected_model_package();
                            }
                        });
                        if let (Some(details), Some(selected_file)) = (
                            selected_package_details.as_ref(),
                            app.selected_mmproj_file_path.as_ref(),
                        ) {
                            if let Some(availability) = details
                                .mmproj_file_availability
                                .iter()
                                .find(|entry| &entry.artifact.relative_path == selected_file)
                            {
                                muted_label(
                                    ui,
                                    &format!(
                                        "MMProj is currently on: {}",
                                        availability
                                            .available_on
                                            .iter()
                                            .map(|node| node.display_name.clone())
                                            .collect::<Vec<_>>()
                                            .join(", ")
                                    ),
                                );
                            }
                        }
                    }
                }

                if app.instance_model_kind == "whisper" {
                    muted_label(
                        ui,
                        "Diarization is now a separate instance. Create a diarization instance on the same owner node if you want /v1/audio/transcriptions requests with diarization enabled.",
                    );
                }

                if let Some(readme) = app.selected_package_readme() {
                    let snippet = readme.lines().take(18).collect::<Vec<_>>().join("\n");
                    if !snippet.trim().is_empty() {
                        ui.add_space(8.0);
                        outlined_card(ui, |ui| {
                            ui.label(egui::RichText::new("Package README preview").strong());
                            render_readme_preview(
                                app,
                                ui,
                                "package-readme-preview",
                                &snippet,
                                180.0,
                            );
                        });
                    }
                }
            }
        });

        if app.selected_model_file_path.is_none() {
            warning_card(
                ui,
                "Pick a model file",
                "Choose the main GGUF or BIN file you want to run from the selected folder.",
            );
            return;
        }
        if app.instance_model_kind == "vision" && app.selected_mmproj_file_path.is_none() {
            warning_card(
                ui,
                "Pick an MMProj file",
                "Vision mode needs both the main model and an mmproj file from the same folder.",
            );
            return;
        }
        ui.add_space(10.0);
        if let Some(model) = app.selected_instance_model_entry() {
            render_model_details(app, ui, &model);
        }
        return;
    }

    card(ui, Some("Loaded runtimes"), |ui| {
        let instances = cluster_instances(app);
        if instances.is_empty() {
            app.selected_instance_id = None;
            ui.label("No runtimes are registered yet.");
            return;
        }
        egui::ScrollArea::vertical()
            .id_salt("cluster-instance-list")
            .max_height(520.0)
            .show(ui, |ui| {
                for (owner_addr, owner_name, instance) in instances {
                    outlined_card(ui, |ui| {
                        let selected = app.selected_instance_id == Some(instance.instance_id)
                            && app.selected_control_addr.as_deref().is_some_and(|selected| {
                                lookup_node_for_addr(app, selected)
                                    .is_some_and(|current| current.control_addr == owner_addr)
                            });
                        if ui
                            .selectable_label(
                                selected,
                                format!("#{} {}", instance.instance_id, instance.name),
                            )
                            .clicked()
                        {
                            app.selected_control_addr = Some(owner_addr.clone());
                            app.selected_instance_id = Some(instance.instance_id);
                            open_instances_loaded_view(app);
                        }
                        ui.horizontal_wrapped(|ui| {
                            muted_label(ui, &owner_name);
                            state_badge(
                                ui,
                                state_label(instance.state),
                                state_fill(instance.state),
                                state_color(instance.state),
                            );
                            state_badge(
                                ui,
                                retention_label(instance.retention_mode),
                                egui::Color32::from_rgb(243, 232, 255),
                                egui::Color32::from_rgb(107, 33, 168),
                            );
                            if instance.retention_mode == RetentionMode::LoadOnDemand {
                                state_badge(
                                    ui,
                                    if instance.load_on_demand_grace_seconds <= 0 {
                                        "Immediate unload".to_string()
                                    } else {
                                        format!("Grace {}s", instance.load_on_demand_grace_seconds)
                                    },
                                    egui::Color32::from_rgb(254, 249, 195),
                                    egui::Color32::from_rgb(133, 77, 14),
                                );
                            }
                            state_badge(
                                ui,
                                format!("Active {}", instance.active_request_count),
                                egui::Color32::from_rgb(224, 242, 254),
                                egui::Color32::from_rgb(14, 116, 144),
                            );
                            if instance.queued_request_count > 0 {
                                state_badge(
                                    ui,
                                    format!("Queued {}", instance.queued_request_count),
                                    egui::Color32::from_rgb(254, 242, 242),
                                    egui::Color32::from_rgb(185, 28, 28),
                                );
                            }
                            state_badge(
                                ui,
                                format!("Slots {}", instance.n_parallel),
                                egui::Color32::from_rgb(236, 252, 203),
                                egui::Color32::from_rgb(63, 98, 18),
                            );
                        });
                        muted_label(ui, &instance.execution_group_id);
                        if !instance.model_path.trim().is_empty() {
                            wrapped_muted_text(ui, &instance.model_path);
                        }
                        render_instance_slot_summary(ui, &instance);
                    });
                }
            });
    });

    if let Some(instance) = app.selected_instance().cloned() {
        ui.add_space(10.0);
        card(ui, Some("Selected runtime"), |ui| {
            ui.label(egui::RichText::new(&instance.name).strong().size(18.0));
            ui.label(format!(
                "Type: {}",
                runtime_kind_label(instance.model_kind.as_dropdown_value())
            ));
            wrapped_muted_text(ui, &format!("Model: {}", instance.model_path));
            if !instance.mmproj_path.trim().is_empty() {
                wrapped_muted_text(ui, &format!("MMProj: {}", instance.mmproj_path));
            }
            ui.label(format!("Execution group: {}", instance.execution_group_id));
            if !instance.rpc_servers.trim().is_empty() {
                wrapped_muted_text(ui, &format!("Remote workers: {}", instance.rpc_servers));
            }
            ui.horizontal_wrapped(|ui| {
                state_badge(
                    ui,
                    state_label(instance.state),
                    state_fill(instance.state),
                    state_color(instance.state),
                );
                state_badge(
                    ui,
                    retention_label(instance.retention_mode),
                    egui::Color32::from_rgb(243, 232, 255),
                    egui::Color32::from_rgb(107, 33, 168),
                );
                if instance.retention_mode == RetentionMode::LoadOnDemand {
                    state_badge(
                        ui,
                        if instance.load_on_demand_grace_seconds <= 0 {
                            "Immediate unload".to_string()
                        } else {
                            format!("Grace {}s", instance.load_on_demand_grace_seconds)
                        },
                        egui::Color32::from_rgb(254, 249, 195),
                        egui::Color32::from_rgb(133, 77, 14),
                    );
                }
                state_badge(
                    ui,
                    format!("Active {}", instance.active_request_count),
                    egui::Color32::from_rgb(224, 242, 254),
                    egui::Color32::from_rgb(14, 116, 144),
                );
                if instance.queued_request_count > 0 {
                    state_badge(
                        ui,
                        format!("Queued {}", instance.queued_request_count),
                        egui::Color32::from_rgb(254, 242, 242),
                        egui::Color32::from_rgb(185, 28, 28),
                    );
                }
                state_badge(
                    ui,
                    format!("Slots {}", instance.n_parallel),
                    egui::Color32::from_rgb(236, 252, 203),
                    egui::Color32::from_rgb(63, 98, 18),
                );
            });
            render_instance_slot_summary(ui, &instance);
            if instance.grace_deadline_unix_ms > 0 {
                ui.label(format!(
                    "Grace deadline: {}",
                    instance.grace_deadline_unix_ms
                ));
            } else if instance.retention_mode == RetentionMode::LoadOnDemand {
                ui.label(if instance.load_on_demand_grace_seconds <= 0 {
                    "Load-on-demand grace: immediate unload".to_string()
                } else {
                    format!(
                        "Load-on-demand grace: {} seconds",
                        instance.load_on_demand_grace_seconds
                    )
                });
            }
            if !instance.last_error.trim().is_empty() {
                warning_card(ui, "Last error", &instance.last_error);
            }
            ui.add_space(10.0);
            ui.horizontal_wrapped(|ui| {
                accent_button(ui, "Load")
                    .clicked()
                    .then(|| app.load_selected());
                secondary_button(ui, "Unload")
                    .clicked()
                    .then(|| app.unload_selected());
                secondary_button(ui, "Toggle retention")
                    .clicked()
                    .then(|| app.toggle_retention_selected());
                warning_button(ui, "Remove")
                    .clicked()
                    .then(|| app.remove_selected());
            });
        });
    }
}

fn open_instances_loaded_view(app: &mut ClusterControllerApp) {
    app.selected_page = ControllerPage::Instances;
    app.instance_creation_open = false;
}

fn render_nodes_page(app: &mut ClusterControllerApp, ui: &mut egui::Ui) {
    ui.label("Use this page only for discovery, pairing approval, and saved reconnects. Live cluster telemetry stays in the sidebar widgets.");
    ui.add_space(8.0);

    card(ui, Some("Discovery and pairing"), |ui| {
        ui.horizontal_wrapped(|ui| {
            if app.discovery_status.active
                && app.discovery_status.mode == crate::protocol::DiscoveryMode::Pairing
            {
                if accent_button(ui, "Stop finding nodes").clicked() {
                    app.stop_pair_discovery();
                }
                if app.discovery_status.expires_unix_ms == 0 {
                    ui.label("Looking for nodes continuously.");
                } else {
                    ui.label(format!(
                        "Looking for nodes for {} more seconds.",
                        discovery_seconds_remaining(app.discovery_status.expires_unix_ms)
                    ));
                }
            } else {
                if accent_button(ui, "Connect and look for nodes").clicked() {
                    app.connect_local_host_and_start_pair_discovery(180);
                }
                muted_label(
                    ui,
                    "Known paired nodes reconnect automatically after launch. Press the button when you want this node to actively announce itself for new pairing.",
                );
            }
        });
        ui.add_space(8.0);
        render_pairing_requests_card(app, ui);
        ui.add_space(12.0);
        render_discovered_nodes_card(app, ui);
    });

    ui.add_space(8.0);

    card(ui, Some("Saved paired nodes"), |ui| {
        let paired = app
            .peers
            .iter()
            .filter(|peer| peer.trusted)
            .cloned()
            .collect::<Vec<_>>();
        if paired.is_empty() {
            ui.label("No paired nodes are saved yet.");
        } else {
            for peer in paired {
                let control_paths =
                    display_control_paths(&peer.control_addr, &peer.known_control_addrs);
                outlined_card(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(egui::RichText::new(&peer.display_name).strong());
                        state_badge(
                            ui,
                            "paired",
                            egui::Color32::from_rgb(220, 252, 231),
                            egui::Color32::from_rgb(22, 101, 52),
                        );
                    });
                    render_control_paths(ui, "Known control paths", &control_paths);
                    ui.horizontal_wrapped(|ui| {
                        if secondary_button(ui, "Inspect").clicked() {
                            app.selected_control_addr = Some(peer.control_addr.clone());
                            let _ = app.refresh_selected_preview();
                        }
                        if warning_button(ui, "Forget").clicked() {
                            app.forget_peer(&peer.control_addr);
                        }
                    });
                });
            }
        }
    });

    ui.add_space(8.0);
    render_multi_node_rpc_settings_card(app, ui);
}

fn render_pairing_requests_card(app: &mut ClusterControllerApp, ui: &mut egui::Ui) {
    outlined_card(ui, |ui| {
        ui.label(egui::RichText::new("Incoming requests").strong());
        if app.pairing_requests.is_empty() {
            muted_label(ui, "No nodes are waiting for approval.");
        } else {
            for request in app.pairing_requests.clone() {
                outlined_card(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(egui::RichText::new(&request.requester_display_name).strong());
                        state_badge(
                            ui,
                            format!("code {}", request.request_code),
                            egui::Color32::from_rgb(224, 242, 254),
                            egui::Color32::from_rgb(14, 116, 144),
                        );
                    });
                    wrapped_muted_text(
                        ui,
                        &format!(
                            "{} | {} | {}",
                            request.requester_control_addr,
                            request.requester_os_name,
                            request.requester_arch
                        ),
                    );
                    ui.horizontal_wrapped(|ui| {
                        if accent_button(ui, "Accept pairing").clicked() {
                            app.accept_pairing_request(&request.request_id);
                        }
                        if secondary_button(ui, "Dismiss").clicked() {
                            app.decline_pairing_request(&request.request_id);
                        }
                    });
                });
            }
        }
    });
}

fn render_discovered_nodes_card(app: &mut ClusterControllerApp, ui: &mut egui::Ui) {
    outlined_card(ui, |ui| {
        ui.label(egui::RichText::new("Discovered nodes").strong());
        let pending = app
            .peers
            .iter()
            .filter(|peer| !peer.trusted)
            .cloned()
            .collect::<Vec<_>>();
        if pending.is_empty() {
            muted_label(ui, "No new nodes are visible right now.");
        } else {
            for peer in pending {
                let control_paths =
                    display_control_paths(&peer.control_addr, &peer.known_control_addrs);
                outlined_card(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(egui::RichText::new(&peer.display_name).strong());
                        muted_label(ui, &peer.os_name);
                        muted_label(ui, &peer.arch);
                    });
                    render_control_paths(ui, "Visible control paths", &control_paths);
                    ui.horizontal_wrapped(|ui| {
                        if accent_button(ui, "Request pairing").clicked() {
                            app.request_pairing(&peer.control_addr);
                        }
                        if secondary_button(ui, "Inspect").clicked() {
                            app.selected_control_addr = Some(peer.control_addr.clone());
                            let _ = app.refresh_selected_preview();
                        }
                    });
                });
            }
        }
    });
}

fn render_playground_page(app: &mut ClusterControllerApp, ui: &mut egui::Ui) {
    ui.label(
        "Run a quick prompt against the selected instance and inspect the performance metrics.",
    );
    ui.add_space(8.0);

    card(ui, Some("Request"), |ui| {
        if let Some(instance) = app.selected_instance() {
            ui.horizontal_wrapped(|ui| {
                ui.label(
                    egui::RichText::new(format!("#{} {}", instance.instance_id, instance.name))
                        .strong(),
                );
                muted_label(ui, &instance.execution_group_id);
                state_badge(
                    ui,
                    state_label(instance.state),
                    state_fill(instance.state),
                    state_color(instance.state),
                );
            });
        } else {
            ui.label("Select a running instance from the Instances page first.");
        }

        ui.horizontal_wrapped(|ui| {
            labeled_i32(ui, "n_predict", &mut app.chat_request.n_predict);
            labeled_f32(ui, "temperature", &mut app.chat_request.temperature);
            labeled_f32(ui, "top_p", &mut app.chat_request.top_p);
            labeled_i32(ui, "top_k", &mut app.chat_request.top_k);
        });
        ui.add_sized(
            [ui.available_width(), 160.0],
            egui::TextEdit::multiline(&mut app.chat_request.prompt)
                .hint_text("Ask the selected instance something useful..."),
        );
        if accent_button(ui, "Run chat").clicked() {
            app.run_chat();
        }
    });

    card(ui, Some("Response"), |ui| {
        if let Some(metrics) = &app.last_chat_metrics {
            ui.label(format_inference_metrics(metrics));
        }
        ui.add_sized(
            [ui.available_width(), 300.0],
            egui::TextEdit::multiline(&mut app.chat_response)
                .interactive(false)
                .hint_text("Response output"),
        );
    });
}

fn render_server_page(app: &mut ClusterControllerApp, ui: &mut egui::Ui) {
    ui.label(
        "Expose managed cluster runtimes over HTTP only when you choose to. In-process bridge and direct mode stay separate.",
    );
    ui.add_space(8.0);

    card(ui, Some("Public API"), |ui| {
        let status = app.server_status.clone();
        ui.checkbox(
            &mut app.server_enabled,
            "Enable HTTP endpoints for named cluster instances",
        );
        muted_label(
            ui,
            "When disabled, cluster instances still run locally and across the cluster, but no HTTP server is exposed.",
        );

        let (mut bind_host, mut bind_port) = parse_bind_addr_parts(&app.server_bind_addr_edit)
            .unwrap_or_else(|| ("127.0.0.1".to_string(), "46310".to_string()));
        let mut bind_host_options = available_public_api_bind_hosts();
        if !bind_host_options
            .iter()
            .any(|option| option.host == bind_host)
        {
            bind_host_options.push(PublicApiBindHostOption {
                host: bind_host.clone(),
                label: bind_host.clone(),
            });
        }
        ui.horizontal_wrapped(|ui| {
            ui.label("Bind host");
            egui::ComboBox::from_id_salt("server-bind-host")
                .selected_text(server_bind_host_option_label(
                    &bind_host,
                    &bind_host_options,
                ))
                .width(adaptive_combo_width(ui, 0.78, 220.0, 520.0))
                .show_ui(ui, |ui| {
                    for option in &bind_host_options {
                        ui.selectable_value(&mut bind_host, option.host.clone(), &option.label);
                    }
                });
            ui.label("Port");
            ui.add_sized([100.0, 24.0], egui::TextEdit::singleline(&mut bind_port));
        });
        app.server_bind_addr_edit = format!("{}:{}", bind_host.trim(), bind_port.trim());
        muted_label(
            ui,
            "Public HTTP stays on 127.0.0.1 or a detected private/link-local network address. Wildcard and public-internet binds are rejected.",
        );
        ui.checkbox(
            &mut app.server_allow_cors,
            "Allow CORS for browser and external web UIs",
        );

        ui.add_space(8.0);
        outlined_card(ui, |ui| {
            ui.label(egui::RichText::new("API key").strong());
            if let Some(status) = &status {
                if status.api_key_present {
                    ui.label(format!(
                        "Stored key: {}",
                        status
                            .api_key_fingerprint
                            .as_deref()
                            .map(|value| format!("configured ({value})"))
                            .unwrap_or_else(|| "configured".to_string())
                    ));
                } else {
                    ui.label("Stored key: none");
                }
            }
            ui.add_sized(
                [ui.available_width(), 24.0],
                egui::TextEdit::singleline(&mut app.server_api_key_edit)
                    .password(true)
                    .hint_text("Paste or generate a new API key"),
            );
            ui.horizontal_wrapped(|ui| {
                if secondary_button(ui, "Generate new key").clicked() {
                    app.generate_server_api_key();
                }
                if secondary_button(ui, "Clear key").clicked() {
                    app.clear_server_api_key();
                }
            });
            if let Some(generated) = app.server_generated_api_key.clone() {
                ui.add_space(6.0);
                ui.label("Generated key, copy it now:");
                let mut preview = generated.clone();
                ui.horizontal_wrapped(|ui| {
                    ui.add_sized(
                        [ui.available_width().min(420.0), 24.0],
                        egui::TextEdit::singleline(&mut preview).desired_width(f32::INFINITY),
                    );
                    if secondary_button(ui, "Copy").clicked() {
                        ui.ctx().copy_text(generated.clone());
                        app.status = "Generated API key copied to clipboard.".to_string();
                    }
                });
            }
            muted_label(
                ui,
                "After saving, the key is hidden again. Clients can use Authorization: Bearer <key> or x-api-key.",
            );
        });
        if app.server_allow_cors {
            ui.add_space(8.0);
            outlined_card(ui, |ui| {
                ui.label(egui::RichText::new("Allowed CORS origins").strong());
                muted_label(
                    ui,
                    "Leave empty to allow any origin. Otherwise add one origin per line, for example http://127.0.0.1:3000",
                );
                ui.add_sized(
                    [ui.available_width(), 84.0],
                    egui::TextEdit::multiline(&mut app.server_allowed_origins_edit),
                );
            });
        }

        ui.add_space(8.0);
        outlined_card(ui, |ui| {
            ui.label(egui::RichText::new("Allowed client IPs / CIDRs").strong());
            muted_label(
                ui,
                "Leave empty to allow any client that can reach the chosen local bind address. Otherwise add one IP or CIDR per line, for example 127.0.0.1, 192.168.1.50, or 192.168.1.0/24.",
            );
            ui.add_sized(
                [ui.available_width(), 84.0],
                egui::TextEdit::multiline(&mut app.server_allowed_client_ips_edit),
            );
        });

        ui.add_space(8.0);
        ui.horizontal_wrapped(|ui| {
            if accent_button(ui, "Apply server settings").clicked() {
                app.apply_server_config();
            }
            if secondary_button(ui, "Refresh server status").clicked() {
                app.refresh_server_status();
            }
        });
    });

    card(ui, Some("Current status"), |ui| {
        if let Some(status) = &app.server_status {
            ui.horizontal_wrapped(|ui| {
                state_badge(
                    ui,
                    if status.running {
                        "running"
                    } else if status.enabled {
                        "stopped"
                    } else {
                        "disabled"
                    },
                    if status.running {
                        egui::Color32::from_rgb(220, 252, 231)
                    } else if status.enabled {
                        egui::Color32::from_rgb(254, 249, 195)
                    } else {
                        egui::Color32::from_rgb(241, 245, 249)
                    },
                    if status.running {
                        egui::Color32::from_rgb(22, 101, 52)
                    } else if status.enabled {
                        egui::Color32::from_rgb(133, 77, 14)
                    } else {
                        egui::Color32::from_rgb(71, 85, 105)
                    },
                );
                state_badge(
                    ui,
                    if status.allow_cors {
                        "CORS on"
                    } else {
                        "CORS off"
                    },
                    egui::Color32::from_rgb(224, 242, 254),
                    egui::Color32::from_rgb(14, 116, 144),
                );
                state_badge(
                    ui,
                    if status.api_key_present {
                        "API key required"
                    } else {
                        "No API key"
                    },
                    egui::Color32::from_rgb(243, 232, 255),
                    egui::Color32::from_rgb(107, 33, 168),
                );
                state_badge(
                    ui,
                    if status.allowed_client_ips.is_empty() {
                        "No IP filter"
                    } else {
                        "IP filter on"
                    },
                    egui::Color32::from_rgb(236, 252, 203),
                    egui::Color32::from_rgb(63, 98, 18),
                );
            });
            ui.label(format!("Configured bind: {}", status.bind_addr));
            ui.label(format!(
                "Reachability: {}",
                server_scope_label(&status.bind_addr)
            ));
            if !status.allowed_origins.is_empty() {
                ui.label(format!(
                    "Allowed origins: {}",
                    status.allowed_origins.join(", ")
                ));
            }
            if !status.allowed_client_ips.is_empty() {
                ui.label(format!(
                    "Allowed client IPs: {}",
                    status.allowed_client_ips.join(", ")
                ));
            }
            if let Some(bound) = &status.effective_bind_addr {
                ui.label(format!("Running on: {}", bound));
            }
            if let Some(advertised) = &status.advertised_addr {
                ui.label(format!("Advertised address: {}", advertised));
            }
            if let Some(error) = &status.last_error {
                warning_card(ui, "Last server error", error);
            }
            muted_label(
                ui,
                "Exposes /v1/models, /v1/responses, /v1/chat/completions, /v1/embeddings, /v1/rerank, and /v1/audio/transcriptions for loaded cluster instances. Use the instance name in the API 'model' field.",
            );
        } else {
            ui.label("Connect the local node to inspect server status.");
        }
    });
}

fn render_local_host_settings_card(app: &mut ClusterControllerApp, ui: &mut egui::Ui) {
    card(ui, Some("Local host"), |ui| {
        ui.set_min_height(196.0);
        let local_control_addrs = app.local_node_control_addrs();
        ui.horizontal_wrapped(|ui| {
            ui.label("Controller endpoint");
            ui.add_sized(
                [adaptive_field_width(ui, 0.62, 220.0, 420.0), 24.0],
                egui::Label::new(egui::RichText::new(app.host.control_addr()).monospace()).wrap(),
            );
            if accent_button(ui, "Reconnect").clicked() {
                app.connect_local_host();
            }
        });
        muted_label(
            ui,
            "A node keeps one identity and advertises every reachable control path it has. The endpoint above is only what this controller is using right now.",
        );
        render_control_paths(ui, "Advertised control paths", &local_control_addrs);
        ui.collapsing("Advanced local bind override", |ui| {
            if ui
                .checkbox(
                    &mut app.local_control_addr_auto,
                    "Reconnect automatically using whatever local control paths are live",
                )
                .changed()
            {
                app.local_control_addr_edit = app.host.control_addr().to_string();
            }
            ui.add_enabled_ui(!app.local_control_addr_auto, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label("Manual controller endpoint");
                    ui.add_sized(
                        [adaptive_field_width(ui, 0.62, 220.0, 420.0), 24.0],
                        egui::TextEdit::singleline(&mut app.local_control_addr_edit),
                    );
                });
            });
            muted_label(
                ui,
                if app.local_control_addr_auto {
                    "Automatic mode ignores stale saved IPs and reconnects through the live address set after restarts, replugging, and DHCP/link-local changes."
                } else {
                    "Manual mode changes only this controller's local dial target. The node still advertises every known control path to other machines."
                },
            );
        });
        wrapped_monospace(
            ui,
            &format!("Models dir: {}", app.local_models_dir().display()),
        );
        if let Some(local_node) = app.local_node_snapshot() {
            ui.label(format!(
                "Public HTTP: {}",
                local_node
                    .advertised_public_api_addr
                    .as_deref()
                    .unwrap_or("not running")
            ));
            if let Some(status) = &local_node.firewall_status {
                ui.label(format!("Firewall: {status}"));
            }
            if local_node.firewall_action_required
                && warning_button(ui, "Configure Firewall").clicked()
            {
                app.configure_local_firewall();
            }
        }
    });
}

fn render_multi_node_rpc_settings_card(app: &mut ClusterControllerApp, ui: &mut egui::Ui) {
    card(ui, Some("Multi-node RPC"), |ui| {
        ui.set_min_height(168.0);
        let changed = ui
            .checkbox(
                &mut app.multi_node_rpc_enabled,
                "Enable multi-node RPC worker on this node",
            )
            .changed();
        muted_label(
            ui,
            "Turn this on only for nodes that should contribute GPUs across machines. Same-machine multi-GPU split does not need the RPC worker.",
        );
        muted_label(
            ui,
            "This is local to this node. If it is off here but still enabled on another paired node, that other node can still contribute GPUs to this machine, but this machine will not contribute GPUs back as an RPC worker.",
        );
        muted_label(
            ui,
            "Turning this off does not disconnect paired nodes or stop the normal control link to them. It only disables this node's RPC worker for future launches. If the embedded RPC host is already running, restart Engine for the change to fully take effect.",
        );
        let restart_required = app.local_rpc_restart_required();
        if let Some(local_node) = app
            .nodes
            .iter()
            .find(|node| node.control_addr == app.host.control_addr())
        {
            let current_worker = if restart_required {
                "still running until restart".to_string()
            } else if local_node.rpc_running {
                local_node
                    .advertised_rpc_endpoint
                    .as_deref()
                    .or(local_node.rpc_endpoint.as_deref())
                    .unwrap_or("running")
                    .to_string()
            } else {
                "off".to_string()
            };
            ui.label(format!("Current worker: {}", current_worker));
            if restart_required {
                ui.add_space(6.0);
                warning_card(
                    ui,
                    "Restart required",
                    "The embedded multi-node RPC host is already running. Restart Engine to fully stop it.",
                );
            }
            ui.add_space(6.0);
            let restart_clicked = if restart_required {
                warning_button(ui, "Restart Engine").clicked()
            } else {
                secondary_button(ui, "Restart Engine").clicked()
            };
            if restart_clicked {
                app.restart_controller(ui.ctx());
            }
        }
        if changed {
            app.apply_multi_node_rpc_setting();
        }
    });
}

fn render_settings_page(app: &mut ClusterControllerApp, ui: &mut egui::Ui) {
    ui.label("Configure the local node host, runtime installation, network exposure, and paired-node reconnects.");
    ui.add_space(8.0);

    card(ui, Some("Appearance"), |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label("Theme");
            let mut changed = false;
            changed |= ui
                .selectable_value(
                    &mut app.theme_preference,
                    ControllerThemePreference::System,
                    "System",
                )
                .changed();
            changed |= ui
                .selectable_value(
                    &mut app.theme_preference,
                    ControllerThemePreference::Dark,
                    "Dark",
                )
                .changed();
            changed |= ui
                .selectable_value(
                    &mut app.theme_preference,
                    ControllerThemePreference::Light,
                    "Light",
                )
                .changed();
            if changed {
                app.apply_theme_preference(ui.ctx());
            }
        });
        let system_theme = ui
            .ctx()
            .system_theme()
            .map(|theme| match theme {
                egui::Theme::Dark => "dark",
                egui::Theme::Light => "light",
            })
            .unwrap_or("unknown");
        muted_label(
            ui,
            &format!("Default follows the OS theme. Current system theme: {system_theme}."),
        );
    });

    card(ui, Some("Device visibility defaults"), |ui| {
        let cpu_changed = ui
            .checkbox(
                &mut app.show_cpu_devices,
                "Enable CPU devices and CPU tuning controls",
            )
            .changed();
        let integrated_changed = ui
            .checkbox(
                &mut app.show_integrated_gpus,
                "Show integrated / shared-memory GPUs on non-macOS nodes",
            )
            .changed();
        muted_label(
            ui,
            "Default mode is GPU-only. CPUs stay hidden unless explicitly enabled. Apple Metal stays visible on macOS.",
        );
        if cpu_changed || integrated_changed {
            app.create_params.allow_cpu = app.show_cpu_devices;
            app.create_params.allow_integrated_gpu = app.show_integrated_gpus;
            let _ = app.refresh_selected_preview();
            app.sync_defaults_from_selected_node();
            app.refresh_placement_candidates();
        }
    });

    card(ui, Some("Runtime"), |ui| {
        let selected_backend = app
            .runtime_install_backends
            .get(app.selected_runtime_install_backend)
            .cloned()
            .unwrap_or_else(|| "auto".to_string());
        let recommended_backend = if app
            .runtime_install_recommendation
            .recommended_backend
            .trim()
            .is_empty()
        {
            selected_backend.clone()
        } else {
            app.runtime_install_recommendation
                .recommended_backend
                .clone()
        };
        ui.horizontal_wrapped(|ui| {
            ui.label("Runtime directory");
            ui.add_sized(
                [ui.available_width() * 0.72, 24.0],
                egui::TextEdit::singleline(&mut app.runtime_dir_edit),
            );
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("Runtime to install");
            egui::ComboBox::from_id_salt("runtime-backend")
                .selected_text(selected_backend.clone())
                .show_ui(ui, |ui| {
                    for (index, backend) in app.runtime_install_backends.iter().enumerate() {
                        ui.selectable_value(
                            &mut app.selected_runtime_install_backend,
                            index,
                            backend,
                        );
                    }
                });
            if accent_button(ui, "Install / Repair").clicked() {
                app.start_runtime_install();
            }
            if crate::runtime_installer::runtime_unblock_supported() {
                if app.runtime_unblock_prompt_active() {
                    if app.runtime_install_in_progress || app.runtime_unblock_in_progress {
                        secondary_button_enabled(ui, "Run Runtime Unblock", false);
                    } else if accent_button(ui, "Run Runtime Unblock").clicked() {
                        app.start_runtime_unblock();
                    }
                } else if secondary_button_enabled(
                    ui,
                    "Run Runtime Unblock",
                    !app.runtime_install_in_progress && !app.runtime_unblock_in_progress,
                )
                .clicked()
                {
                    app.start_runtime_unblock();
                }
            }
            if secondary_button(ui, "Reconnect runtime").clicked() {
                app.connect_local_host();
            }
        });
        muted_label(
            ui,
            &format!(
                "Recommended: {}. {}",
                recommended_backend.to_ascii_uppercase(),
                app.runtime_install_recommendation.recommended_reason
            ),
        );
        if let Some(gpu_label) = app
            .runtime_install_recommendation
            .detected_gpu_label
            .as_deref()
        {
            muted_label(ui, &format!("Detected graphics card: {gpu_label}"));
        }
        if let Some(notice) = app
            .runtime_install_recommendation
            .cuda_candidate_notice
            .as_deref()
        {
            muted_label(ui, notice);
        }
        if let Some(installed_backend) = app
            .runtime_install_recommendation
            .installed_backend
            .as_deref()
        {
            muted_label(
                ui,
                &format!(
                    "Installed runtime backend: {}",
                    installed_backend.to_ascii_uppercase()
                ),
            );
            if !selected_backend.eq_ignore_ascii_case(installed_backend) {
                warning_card(
                    ui,
                    "Switching runtime backend",
                    "If you want a different runtime than the one already installed, close the app, open the runtime folder, delete the engine folder, then reinstall the new runtime.",
                );
            }
        }
        if app.runtime_missing.is_empty() {
            if app.runtime_unblock_prompt_active() {
                ui.label(
                    "Runtime looks complete, but still needs Runtime Unblock before first use.",
                );
            } else {
                ui.label("Runtime looks complete.");
            }
        } else {
            ui.colored_label(
                egui::Color32::from_rgb(153, 27, 27),
                "Engine runtime missing or incomplete.",
            );
            warning_card(ui, "Runtime issues", &app.runtime_missing.join("\n"));
        }
        if app.runtime_unblock_prompt_active() {
            warning_card(
                ui,
                "Runtime unblock recommended",
                "Engine just installed or repaired the runtime. Run Runtime Unblock once before first use. On Windows this clears Mark of the Web from downloaded runtime files. On macOS it clears quarantine and restores executable bits for bundled tools.",
            );
        } else if crate::runtime_installer::runtime_unblock_supported() {
            muted_label(
                ui,
                "Run Runtime Unblock if Windows or macOS blocked a downloaded runtime after install.",
            );
        }
        if let Some(status) = &app.runtime_install_status {
            ui.label(status);
        }
        if app.runtime_unblock_in_progress {
            ui.horizontal_wrapped(|ui| {
                ui.spinner();
                muted_label(
                    ui,
                    app.runtime_unblock_status
                        .as_deref()
                        .unwrap_or("Running runtime unblock..."),
                );
            });
        } else if let Some(status) = &app.runtime_unblock_status {
            ui.label(status);
        }
    });

    render_local_host_settings_card(app, ui);
}

fn render_about_page(app: &mut ClusterControllerApp, ui: &mut egui::Ui) {
    ui.label(format!(
        "Openresearchtools-Engine version: {}",
        app.controller_version_label
    ));
    ui.add_space(4.0);
    ui.label("Openresearchtools-Engine is the desktop controller for ENGINE's local and multi-node runtime stack.");
    ui.add_space(4.0);
    ui.label(
        "It lets you discover and pair machines, see which devices and models are available on each node, create named runtimes on any connected system, and keep one stable control point even when the actual runtime is running somewhere else in the cluster.",
    );
    ui.label(
        "Engine is based on the llama.cpp runtime family together with ENGINE's own native bridge and clustering layers. That includes remote runtime ownership, cross-node workers, model transfer between nodes, named-instance HTTP routing, native audio and transcription paths, and mixed-device orchestration across CUDA, Vulkan, and Metal systems.",
    );
    ui.label(
        "Any connected node can act as the initiator, the runtime owner, or a worker. Direct Thunderbolt or USB4 link-local connections are preferred where available because they provide the best practical speeds for model transfer and cross-node traffic. CPU execution is available by enabling CPU devices in Settings, but it is generally not recommended for clustered tensor-model workloads, is often untested, and is dramatically slower than GPU-backed execution.",
    );
    ui.label("Bug reports, issues, and development updates live in the ENGINE repository:");
    ui.hyperlink("https://github.com/openresearchtools/engine");

    ui.add_space(12.0);
    card(ui, Some("Documents"), |ui| {
        ui.label(
            "The main ENGINE manual, license, and bundled third-party notices and licenses are embedded into the app at build time.",
        );
        ui.add_space(8.0);
        ui.horizontal_wrapped(|ui| {
            if subtab_button(
                ui,
                "Engine License",
                app.selected_about_document == AboutDocument::EngineLicense,
            )
            .clicked()
            {
                app.selected_about_document = AboutDocument::EngineLicense;
            }
            if subtab_button(
                ui,
                "Third-Party Notices",
                app.selected_about_document == AboutDocument::ThirdPartyNotices,
            )
            .clicked()
            {
                app.selected_about_document = AboutDocument::ThirdPartyNotices;
            }
            if subtab_button(
                ui,
                "Third-Party Licenses",
                app.selected_about_document == AboutDocument::ThirdPartyLicenses,
            )
            .clicked()
            {
                app.selected_about_document = AboutDocument::ThirdPartyLicenses;
            }
            if subtab_button(
                ui,
                "Manual",
                app.selected_about_document == AboutDocument::Manual,
            )
            .clicked()
            {
                app.selected_about_document = AboutDocument::Manual;
            }
        });
        ui.add_space(10.0);
        match app.selected_about_document {
            AboutDocument::EngineLicense => {
                ui.label(egui::RichText::new("Engine License").strong().size(16.0));
                ui.add_space(6.0);
                render_about_document(app, ui, ENGINE_LICENSE_TEXT);
            }
            AboutDocument::ThirdPartyNotices => {
                ui.label(
                    egui::RichText::new("Third-Party Notices")
                        .strong()
                        .size(16.0),
                );
                ui.add_space(6.0);
                render_about_document(app, ui, THIRD_PARTY_NOTICES_TEXT);
            }
            AboutDocument::ThirdPartyLicenses => {
                ui.label(
                    egui::RichText::new("Third-Party Licenses")
                        .strong()
                        .size(16.0),
                );
                ui.add_space(6.0);
                render_about_document(app, ui, THIRD_PARTY_LICENSES_TEXT);
            }
            AboutDocument::Manual => {
                ui.label(egui::RichText::new("Manual").strong().size(16.0));
                ui.add_space(6.0);
                render_readme_preview(app, ui, "about-engine-manual", ENGINE_MANUAL_TEXT, 0.0);
            }
        }
    });
}

#[allow(unreachable_code)]
fn render_model_details(
    app: &mut ClusterControllerApp,
    ui: &mut egui::Ui,
    model: &ManagedModelEntry,
) {
    app.create_params.n_parallel = app.create_params.n_parallel.max(1);
    let detected_metadata = app.selected_model_metadata();
    if let Some(layer_count) = detected_metadata
        .as_ref()
        .and_then(|metadata| metadata.block_count)
    {
        if app.create_params.n_gpu_layers > layer_count as i32 {
            app.create_params.n_gpu_layers = layer_count as i32;
        }
        for row in &mut app.create_params.manual_device_allocations {
            if row.layer_count > layer_count {
                row.layer_count = layer_count;
            }
        }
    }
    let runtime_estimate = app.selected_runtime_vram_estimate();
    let targets = placement_targets(app);
    let auto_selected = app.create_params.execution_group_id == "cluster:auto"
        || app.create_params.execution_group_id.trim().is_empty();
    let auto_target_label = if model.single_device_only {
        "Automatic single GPU across visible nodes"
    } else {
        "Automatic best visible target"
    };
    let selected_target = if auto_selected {
        targets.first().cloned()
    } else {
        targets
            .iter()
            .find(|target| {
                app.create_params.preferred_owner_control_addr.as_deref()
                    == Some(target.owner_control_addr.as_str())
                    && app.create_params.execution_group_id == target.execution_group_id
                    && app.create_params.rpc_servers.clone().unwrap_or_default()
                        == target.rpc_servers
            })
            .cloned()
            .or_else(|| targets.first().cloned())
    };
    ui.label(egui::RichText::new(&model.display_name).strong().size(18.0));
    ui.horizontal_wrapped(|ui| {
        state_badge(
            ui,
            task_label(model.task),
            egui::Color32::from_rgb(224, 242, 254),
            egui::Color32::from_rgb(14, 116, 144),
        );
        state_badge(
            ui,
            &model.family,
            egui::Color32::from_rgb(243, 232, 255),
            egui::Color32::from_rgb(107, 33, 168),
        );
        if model.supports_vision() {
            state_badge(
                ui,
                "vision",
                egui::Color32::from_rgb(255, 237, 213),
                egui::Color32::from_rgb(154, 52, 18),
            );
        }
        if model.supports_diarization() {
            state_badge(
                ui,
                "diarization",
                egui::Color32::from_rgb(220, 252, 231),
                egui::Color32::from_rgb(22, 101, 52),
            );
        }
        state_badge(
            ui,
            if model.single_device_only {
                "single GPU"
            } else {
                "split-capable"
            },
            if model.single_device_only {
                egui::Color32::from_rgb(254, 249, 195)
            } else {
                egui::Color32::from_rgb(219, 234, 254)
            },
            if model.single_device_only {
                egui::Color32::from_rgb(133, 77, 14)
            } else {
                egui::Color32::from_rgb(30, 64, 175)
            },
        );
    });
    ui.label(format!("Model path: {}", model.model_path));
    if let Some(mmproj) = &model.mmproj_path {
        ui.label(format!("MMProj: {mmproj}"));
    }
    if let Some(path) = &model.diarization_model_path {
        ui.label(format!("Diarization: {path}"));
    }

    render_integrated_model_details(
        app,
        ui,
        model,
        &targets,
        auto_selected,
        auto_target_label,
        selected_target.as_ref(),
        detected_metadata.as_ref(),
        runtime_estimate.as_ref(),
    );
    return;

    card(ui, Some("Recommended setup"), |ui| {
        ui.label(if model.single_device_only {
            "This runtime stays on one GPU on one allowed node. Automatic mode picks the best single discrete GPU without using CPU devices."
        } else {
            "Automatic mode is math-first: one discrete GPU if it fits, then same-host multi-GPU split, then multi-node split across trusted GPU nodes."
        });
        ui.label(format!(
            "Allowed nodes right now: {} | Runtime slots: {}",
            if app.allowed_control_addrs.is_empty() {
                app.default_allowed_node_addrs().len()
            } else {
                app.allowed_control_addrs.len()
            },
            app.create_params.n_parallel.max(1)
        ));
        let available_nodes = model_available_node_labels(app, model);
        if !available_nodes.is_empty() {
            muted_label(ui, &format!("Available on: {}", available_nodes.join(", ")));
        }
        if model.task == ManagedModelTask::Transcription {
            muted_label(
                ui,
                "Bridge, managed HTTP, and tray-host scheduling all route this family to one allocated GPU runtime. Split execution is intentionally disabled for these audio families.",
            );
            if model.supports_diarization() {
                muted_label(
                    ui,
                    "If diarization is enabled at request time, the API looks for a separate diarization instance on the same owner node and uses that companion model there.",
                );
            }
        } else {
            muted_label(
                ui,
                "Pinned placement obeys exactly the selected owner node and execution target. Automatic placement only falls back to broader splits when the fit math says it is needed.",
            );
        }
    });

    let targets = placement_targets(app);
    card(
        ui,
        Some(if model.single_device_only {
            "Choose a node"
        } else {
            "Where this model can run"
        }),
        |ui| {
            let available_addrs = model
                .allowed_control_addrs
                .clone()
                .unwrap_or_else(|| app.default_allowed_node_addrs().into_iter().collect());
            let mut rows = available_addrs
                .into_iter()
                .filter_map(|addr| lookup_node_for_addr(app, &addr).cloned())
                .collect::<Vec<_>>();
            rows.sort_by(|lhs, rhs| lhs.node.display_name.cmp(&rhs.node.display_name));
            rows.dedup_by(|lhs, rhs| lhs.control_addr == rhs.control_addr);

            if rows.is_empty() {
                muted_label(
                    ui,
                    "No reachable GPU nodes are visible for this model yet. Connect the local node, trust any peers, then refresh the cluster.",
                );
                return;
            }

            for node in rows {
                let best_target =
                    best_target_for_owner_control_addr(&targets, &node.control_addr).cloned();
                outlined_card(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(egui::RichText::new(&node.node.display_name).strong());
                        if let Some(target) = &best_target {
                            state_badge(
                                ui,
                                if target.ready_now {
                                    "ready now"
                                } else if target.requires_eviction {
                                    "eviction needed"
                                } else if target.rpc_servers.trim().is_empty() {
                                    "single GPU visible"
                                } else {
                                    "split target visible"
                                },
                                if target.ready_now {
                                    egui::Color32::from_rgb(220, 252, 231)
                                } else if target.requires_eviction {
                                    egui::Color32::from_rgb(255, 237, 213)
                                } else if target.rpc_servers.trim().is_empty() {
                                    egui::Color32::from_rgb(254, 249, 195)
                                } else {
                                    egui::Color32::from_rgb(255, 237, 213)
                                },
                                if target.ready_now {
                                    egui::Color32::from_rgb(22, 101, 52)
                                } else if target.requires_eviction {
                                    egui::Color32::from_rgb(154, 52, 18)
                                } else if target.rpc_servers.trim().is_empty() {
                                    egui::Color32::from_rgb(133, 77, 14)
                                } else {
                                    egui::Color32::from_rgb(154, 52, 18)
                                },
                            );
                        } else {
                            state_badge(
                                ui,
                                "no visible target",
                                egui::Color32::from_rgb(254, 226, 226),
                                egui::Color32::from_rgb(153, 27, 27),
                            );
                        }
                    });
                    if let Some(target) = &best_target {
                        muted_label(
                            ui,
                            &format!(
                                "{}{}",
                                if model.single_device_only {
                                    "Best single-GPU target here: "
                                } else {
                                    "Best visible target here: "
                                },
                                target.title
                            ),
                        );
                    } else {
                        muted_label(
                            ui,
                            "This node is allowed for the model, but no eligible GPU target is visible with the current filters.",
                        );
                    }
                    ui.horizontal_wrapped(|ui| {
                        if secondary_button(ui, "Automatic on this node").clicked() {
                            app.allowed_control_addrs.clear();
                            ClusterControllerApp::set_addr_selection_for_node(
                                &mut app.allowed_control_addrs,
                                &node,
                                true,
                            );
                            set_auto_placement_target(app);
                            app.refresh_placement_candidates();
                        }
                        if let Some(target) = &best_target {
                            if accent_button(
                                ui,
                                if model.single_device_only {
                                    "Pin best GPU here"
                                } else {
                                    "Pin best target here"
                                },
                            )
                            .clicked()
                            {
                                apply_placement_target(app, target);
                                app.refresh_placement_candidates();
                            }
                            if secondary_button_enabled(
                                ui,
                                "Load here now",
                                placement_target_can_load(target),
                            )
                            .clicked()
                            {
                                apply_placement_target(app, target);
                                app.schedule_instance_cluster();
                                open_instances_loaded_view(app);
                            }
                        }
                    });
                });
            }
        },
    );

    ui.add_space(10.0);
    ui.horizontal_wrapped(|ui| {
        ui.label("Instance name");
        if ui
            .text_edit_singleline(&mut app.create_params.name)
            .changed()
        {
            app.sync_instance_name_edit_state();
        }
        ui.label("Retention");
        egui::ComboBox::from_id_salt("retention-mode")
            .selected_text(retention_label(app.create_params.retention_mode))
            .show_ui(ui, |ui| {
                ui.selectable_value(
                    &mut app.create_params.retention_mode,
                    RetentionMode::KeepLoaded,
                    "keep loaded",
                );
                ui.selectable_value(
                    &mut app.create_params.retention_mode,
                    RetentionMode::LoadOnDemand,
                    "load on demand",
                );
            });
        render_load_on_demand_grace_editor(ui, app);
    });

    card(ui, Some("Current run summary"), |ui| {
        let allowed_nodes = if app.allowed_control_addrs.is_empty() {
            app.default_allowed_node_addrs()
        } else {
            app.allowed_control_addrs.clone()
        };
        let allowed_labels = app
            .nodes
            .iter()
            .filter(|node| ClusterControllerApp::addr_selection_contains_node(&allowed_nodes, node))
            .map(|node| node.node.display_name.clone())
            .collect::<Vec<_>>();
        let automatic_mode = app.create_params.execution_group_id == "cluster:auto"
            || app.create_params.execution_group_id.trim().is_empty();
        ui.label(if automatic_mode {
            "Mode: Automatic (math-first)"
        } else {
            "Mode: Pinned exact target"
        });
        ui.label(format!(
            "Retention: {}{} | reserved request slots: {}",
            retention_label(app.create_params.retention_mode),
            load_on_demand_grace_summary_suffix(
                app.create_params.retention_mode,
                app.create_params.load_on_demand_grace_seconds,
            ),
            app.create_params.n_parallel.max(1)
        ));
        ui.label(if allowed_labels.is_empty() {
            "Allowed nodes: none selected yet".to_string()
        } else {
            format!("Allowed nodes: {}", allowed_labels.join(", "))
        });
        if let Some(plan) = &app.last_plan {
            ui.label(format!(
                "Latest plan: {} on {}",
                if plan.display_label.trim().is_empty() {
                    plan.execution_group_id.as_str()
                } else {
                    plan.display_label.as_str()
                },
                plan.owner_display_name
            ));
            muted_label(
                ui,
                &format!(
                    "Needs {} free and currently sees {} on the chosen target.",
                    format_mib_from_bytes(plan.estimated_required_bytes),
                    format_mib_from_bytes(plan.estimated_group_free_bytes)
                ),
            );
        } else if automatic_mode {
            muted_label(
                ui,
                "No plan has been generated yet. Automatic mode will prefer one discrete GPU first, then same-host split, then multi-node split only if needed.",
            );
        } else {
            muted_label(
                ui,
                "No plan has been generated yet. Pinned mode will load exactly on the selected target or fail clearly.",
            );
        }
    });

    let auto_target_label = if model.single_device_only {
        "Automatic single GPU across allowed nodes"
    } else {
        "Automatic across allowed GPU nodes"
    };
    let auto_selected = app.create_params.execution_group_id == "cluster:auto"
        || app.create_params.execution_group_id.trim().is_empty();
    let pinned_target_label = if targets.is_empty() {
        "No GPU targets available".to_string()
    } else {
        selected_target_label(app, &targets, auto_target_label)
    };
    card(ui, Some("Best GPU targets right now"), |ui| {
        if targets.is_empty() {
            muted_label(
                ui,
                "No eligible GPU targets are visible yet. Check node trust, runtime install, and allowed-node filters.",
            );
            return;
        }

        if auto_selected {
            muted_label(
                ui,
                "Automatic mode will pick the first valid target here unless the fit math forces a broader split.",
            );
        } else {
            muted_label(
                ui,
                "Pinned mode can be switched back to Automatic at any time. The cards below are the visible choices right now.",
            );
        }

        let spotlight_targets = highlighted_placement_targets(&targets);
        for (index, target) in spotlight_targets.iter().enumerate() {
            let selected = app.create_params.preferred_owner_control_addr.as_deref()
                == Some(target.owner_control_addr.as_str())
                && app.create_params.execution_group_id == target.execution_group_id
                && app.create_params.rpc_servers.clone().unwrap_or_default() == target.rpc_servers;
            outlined_card(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label(egui::RichText::new(&target.title).strong());
                    if index == 0 {
                        state_badge(
                            ui,
                            "best fit now",
                            egui::Color32::from_rgb(220, 252, 231),
                            egui::Color32::from_rgb(22, 101, 52),
                        );
                    }
                    if selected {
                        state_badge(
                            ui,
                            "pinned",
                            egui::Color32::from_rgb(224, 242, 254),
                            egui::Color32::from_rgb(14, 116, 144),
                        );
                    }
                    if !target.rpc_servers.trim().is_empty() {
                        state_badge(
                            ui,
                            "multi-node",
                            egui::Color32::from_rgb(255, 237, 213),
                            egui::Color32::from_rgb(154, 52, 18),
                        );
                    } else if target.execution_group_id.contains("split") {
                        state_badge(
                            ui,
                            "same-host split",
                            egui::Color32::from_rgb(243, 232, 255),
                            egui::Color32::from_rgb(107, 33, 168),
                        );
                    } else {
                        state_badge(
                            ui,
                            "single GPU",
                            egui::Color32::from_rgb(254, 249, 195),
                            egui::Color32::from_rgb(133, 77, 14),
                        );
                    }
                    if target.ready_now {
                        state_badge(
                            ui,
                            "ready now",
                            egui::Color32::from_rgb(220, 252, 231),
                            egui::Color32::from_rgb(22, 101, 52),
                        );
                    } else if target.requires_eviction {
                        state_badge(
                            ui,
                            "eviction needed",
                            egui::Color32::from_rgb(255, 237, 213),
                            egui::Color32::from_rgb(154, 52, 18),
                        );
                    } else if target.estimated_required_bytes > 0 {
                        state_badge(
                            ui,
                            "insufficient free memory",
                            egui::Color32::from_rgb(254, 226, 226),
                            egui::Color32::from_rgb(153, 27, 27),
                        );
                    }
                });
                muted_label(ui, &target.subtitle);
                ui.horizontal_wrapped(|ui| {
                    if accent_button(ui, "Pin this target").clicked() {
                        apply_placement_target(app, target);
                        app.refresh_placement_candidates();
                    }
                    if secondary_button_enabled(
                        ui,
                        "Load here now",
                        placement_target_can_load(target),
                    )
                    .clicked()
                    {
                        apply_placement_target(app, target);
                        app.schedule_instance_cluster();
                        open_instances_loaded_view(app);
                    }
                    if !auto_selected && secondary_button(ui, "Use automatic").clicked() {
                        set_auto_placement_target(app);
                        app.refresh_placement_candidates();
                    }
                });
            });
        }
    });
    card(ui, Some("Placement mode"), |ui| {
        ui.horizontal_wrapped(|ui| {
            if ui
                .selectable_label(auto_selected, "Automatic (recommended)")
                .clicked()
            {
                set_auto_placement_target(app);
            }
            if ui
                .selectable_label(!auto_selected, "Pin exact GPU target")
                .clicked()
                && !targets.is_empty()
            {
                if auto_selected {
                    apply_placement_target(app, &targets[0]);
                }
            }
        });
        ui.add_space(6.0);
        if auto_selected {
            ui.label(if model.single_device_only {
                "The scheduler will pick one discrete GPU on the best allowed node. CPU and split placement stay off for this model family."
            } else {
                "The scheduler will do the math-first choice: one discrete GPU if it fits, then same-host split, then multi-node split only when needed."
            });
            let recommendation = targets
                .first()
                .map(|target| {
                    if placement_target_can_load(target) {
                        format!("Best visible target right now: {}", target.title)
                    } else {
                        format!(
                            "No visible target is loadable right now. Current best visible shape: {}",
                            target.title
                        )
                    }
                })
                .unwrap_or_else(|| {
                    "No eligible GPU targets are visible yet. Check node trust and runtime status."
                        .to_string()
                });
            muted_label(ui, &recommendation);
        } else {
            ui.label(if model.single_device_only {
                "Pinned mode keeps this runtime on one exact GPU target."
            } else {
                "Pinned mode obeys the exact owner node and execution target you choose below."
            });
            egui::ComboBox::from_id_salt("placement-target")
                .selected_text(pinned_target_label)
                .width(520.0)
                .show_ui(ui, |ui| {
                    for target in &targets {
                        let selected = app.create_params.preferred_owner_control_addr.as_deref()
                            == Some(target.owner_control_addr.as_str())
                            && app.create_params.execution_group_id == target.execution_group_id
                            && app.create_params.rpc_servers.clone().unwrap_or_default()
                                == target.rpc_servers;
                        if ui
                            .selectable_label(selected, &target.title)
                            .on_hover_text(&target.subtitle)
                            .clicked()
                        {
                            apply_placement_target(app, target);
                        }
                    }
                });
            egui::ScrollArea::vertical()
                .id_salt("placement-target-list")
                .max_height(220.0)
                .show(ui, |ui| {
                    for target in &targets {
                        let selected = app.create_params.preferred_owner_control_addr.as_deref()
                            == Some(target.owner_control_addr.as_str())
                            && app.create_params.execution_group_id == target.execution_group_id
                            && app.create_params.rpc_servers.clone().unwrap_or_default()
                                == target.rpc_servers;
                        outlined_card(ui, |ui| {
                            if ui.selectable_label(selected, &target.title).clicked() {
                                apply_placement_target(app, target);
                            }
                            muted_label(ui, &target.subtitle);
                        });
                    }
                });
        }
    });

    card(ui, Some("Allowed nodes"), |ui| {
        let default_allowed = app.default_allowed_node_addrs();
        if app.allowed_control_addrs.is_empty() {
            app.allowed_control_addrs = default_allowed.clone();
        }
        for node in &app.nodes.clone() {
            let mut enabled =
                ClusterControllerApp::addr_selection_contains_node(&app.allowed_control_addrs, node);
            let visible_groups = filtered_execution_groups_for_node(app, node, None)
                .into_iter()
                .filter(|group| group.id != "cluster:auto")
                .count();
            let device_preview = filtered_devices_for_node(
                app,
                node,
                app.telemetry_for_control_addr(&node.control_addr),
            )
            .into_iter()
            .filter(|device| !is_rpc_device(device))
            .map(|device| device_display_name_for_ui(app, node, &device))
            .take(2)
            .collect::<Vec<_>>()
            .join(" + ");
            let description = format!(
                "{} | {} target{}{}{}",
                node.node.display_name,
                visible_groups,
                if visible_groups == 1 { "" } else { "s" },
                if node.rpc_running {
                    " | split worker ready"
                } else if visible_groups > 0 {
                    " | single-device ready"
                } else {
                    ""
                },
                if device_preview.is_empty() {
                    String::new()
                } else {
                    format!(" | {device_preview}")
                }
            );
            let response = ui.checkbox(&mut enabled, description);
            let response = response.on_hover_text(
                display_control_paths(&node.control_addr, &node.known_control_addrs).join("\n"),
            );
            if response.changed() {
                ClusterControllerApp::set_addr_selection_for_node(
                    &mut app.allowed_control_addrs,
                    node,
                    enabled,
                );
                app.refresh_placement_candidates();
            }
        }
        ui.horizontal_wrapped(|ui| {
            if secondary_button(ui, "Use all reachable").clicked() {
                app.allowed_control_addrs = default_allowed;
                app.refresh_placement_candidates();
            }
            if secondary_button(ui, "Use current node only").clicked() {
                app.allowed_control_addrs.clear();
                if let Some(selected) = app
                    .create_params
                    .preferred_owner_control_addr
                    .as_ref()
                    .or(app.selected_control_addr.as_ref())
                {
                    if let Some(node) = lookup_node_for_addr(app, selected).cloned() {
                        ClusterControllerApp::set_addr_selection_for_node(
                            &mut app.allowed_control_addrs,
                            &node,
                            true,
                        );
                    } else {
                        app.allowed_control_addrs.insert(selected.to_string());
                    }
                }
                app.refresh_placement_candidates();
            }
        });
    });

    card(ui, Some("GPU runtime"), |ui| {
        ui.horizontal_wrapped(|ui| {
            labeled_i32(ui, "n_ctx", &mut app.create_params.n_ctx);
            labeled_i32(ui, "n_batch", &mut app.create_params.n_batch);
            labeled_i32(ui, "n_ubatch", &mut app.create_params.n_ubatch);
            labeled_i32(ui, "gpu_layers", &mut app.create_params.n_gpu_layers);
        });
        ui.add_space(6.0);
        ui.horizontal_wrapped(|ui| {
            ui.label("Parallel slots");
            for preset in [1, 2, 3] {
                let selected = app.create_params.n_parallel == preset;
                if ui
                    .selectable_label(selected, format!("{preset}"))
                    .on_hover_text(match preset {
                        1 => "Lowest VRAM pressure. Best for the largest runtimes.",
                        2 => "Keep one extra request slot warm if VRAM allows it.",
                        _ => "Reserve room for up to three active request contexts.",
                    })
                    .clicked()
                {
                    app.create_params.n_parallel = preset;
                }
            }
            ui.add(
                egui::DragValue::new(&mut app.create_params.n_parallel)
                    .range(1..=8)
                    .speed(1),
            );
        });
        muted_label(
            ui,
            "GPU layers defaults to -1. Slots reserve warm request contexts for the same loaded model; 1 is safest for the largest models, 2-3 only when VRAM headroom is real. True cross-node pipeline scheduling remains a later runtime step.",
        );
    });

    if app.show_cpu_devices {
        card(ui, Some("CPU and host tuning"), |ui| {
            ui.horizontal_wrapped(|ui| {
                labeled_i32(ui, "threads", &mut app.create_params.n_threads);
                labeled_i32(ui, "threads_batch", &mut app.create_params.n_threads_batch);
            });
        });
    }

    ui.add_space(10.0);
    ui.horizontal_wrapped(|ui| {
        if accent_button(ui, "Plan placement").clicked() {
            app.plan_instance_cluster();
        }
        if accent_button(ui, "Load now").clicked() {
            app.schedule_instance_cluster();
            open_instances_loaded_view(app);
        }
        if secondary_button(
            ui,
            if app.show_advanced_instance_editor {
                "Hide advanced manual controls"
            } else {
                "Show advanced manual controls"
            },
        )
        .clicked()
        {
            app.show_advanced_instance_editor = !app.show_advanced_instance_editor;
        }
    });

    if let Some(plan) = &app.last_plan {
        ui.add_space(8.0);
        outlined_card(ui, |ui| {
            ui.label(egui::RichText::new("Placement preview").strong().size(15.0));
            let placement_label = if plan.display_label.trim().is_empty() {
                plan.execution_group_id.clone()
            } else {
                plan.display_label.clone()
            };
            ui.label(format!(
                "{} on {} via {}",
                placement_strategy_label(plan.strategy),
                plan.owner_display_name,
                placement_label
            ));
            ui.label(format!(
                "Required {} | free {} | ready {} | reuse {} | eviction {}",
                format_mib_from_bytes(plan.estimated_required_bytes),
                format_mib_from_bytes(plan.estimated_group_free_bytes),
                yes_no(plan.ready_now),
                plan.reusable_instance_id
                    .map(|value: i64| value.to_string())
                    .unwrap_or_else(|| "<none>".to_string()),
                yes_no(plan.requires_eviction),
            ));
            if !plan.rpc_servers.is_empty() {
                ui.label(format!("Remote workers: {}", plan.rpc_servers));
            }
        });
    }

    if app.show_advanced_instance_editor {
        ui.add_space(10.0);
        card(ui, Some("Advanced manual creation"), |ui| {
            ui.label("Direct-path and explicit remote worker controls stay available for debugging and power users.");
            ui.horizontal_wrapped(|ui| {
                ui.label("Model path");
                ui.add_sized(
                    [ui.available_width() * 0.7, 24.0],
                    egui::TextEdit::singleline(&mut app.create_params.model_path),
                );
            });
            let mmproj = app
                .create_params
                .mmproj_path
                .get_or_insert_with(String::new);
            ui.horizontal_wrapped(|ui| {
                ui.label("MMProj");
                ui.add_sized(
                    [ui.available_width() * 0.7, 24.0],
                    egui::TextEdit::singleline(mmproj),
                );
            });
            ui.label("Remote worker preview");
            for node in &app.nodes.clone() {
                if app.selected_control_addr.as_deref().is_some_and(|selected| {
                    lookup_node_for_addr(app, selected)
                        .is_some_and(|current| current.control_addr == node.control_addr)
                }) {
                    continue;
                }
                let mut enabled =
                    ClusterControllerApp::addr_selection_contains_node(&app.selected_rpc_peer_addrs, node);
                let label = format!(
                    "{} | {}",
                    node.node.display_name,
                    if node.rpc_running {
                        "split worker ready"
                    } else {
                        "rpc unavailable"
                    }
                );
                let hover = format!(
                    "Control paths:\n{}\n\nRPC endpoint: {}",
                    display_control_paths(&node.control_addr, &node.known_control_addrs).join("\n"),
                    node.advertised_rpc_endpoint
                        .as_deref()
                        .or(node.rpc_endpoint.as_deref())
                        .unwrap_or("unavailable")
                );
                if ui.checkbox(&mut enabled, label).on_hover_text(hover).changed() {
                    ClusterControllerApp::set_addr_selection_for_node(
                        &mut app.selected_rpc_peer_addrs,
                        node,
                        enabled,
                    );
                    if let Err(err) = app.refresh_selected_preview() {
                        app.status = err;
                    }
                }
            }
            ui.horizontal_wrapped(|ui| {
                if secondary_button(ui, "Create on selected node").clicked() {
                    app.create_instance();
                    open_instances_loaded_view(app);
                }
                if secondary_button(ui, "Select all remote workers").clicked() {
                    app.select_all_remote_rpc_peers();
                }
                if secondary_button(ui, "Clear remote workers").clicked() {
                    app.clear_remote_rpc_peers();
                }
            });
        });
    }
}

fn render_integrated_model_details_legacy(
    app: &mut ClusterControllerApp,
    ui: &mut egui::Ui,
    model: &ManagedModelEntry,
    targets: &[PlacementTargetView],
    auto_selected: bool,
    auto_target_label: &str,
    selected_target: Option<&PlacementTargetView>,
    detected_metadata: Option<&ModelFileMetadata>,
    runtime_estimate: Option<&RuntimeVramEstimate>,
) {
    card(ui, Some("Launch setup"), |ui| {
        ui.label(
            "The best visible placement is preselected. The target dropdown still exposes every visible single-GPU, same-host split, and multi-node combination so you can override it directly.",
        );
        ui.add_space(8.0);

        ui.horizontal_wrapped(|ui| {
            ui.label("Preset");
            let previous_preset = app.selected_instance_preset_name.clone();
            egui::ComboBox::from_id_salt("instance-preset-picker")
                .selected_text(
                    app.selected_instance_preset_name
                        .clone()
                        .unwrap_or_else(|| "No preset".to_string()),
                )
                .width(adaptive_combo_width(ui, 0.34, 200.0, 320.0))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut app.selected_instance_preset_name, None, "No preset");
                    for preset in &app.instance_presets {
                        ui.selectable_value(
                            &mut app.selected_instance_preset_name,
                            Some(preset.name.clone()),
                            &preset.name,
                        );
                    }
                });
            if previous_preset != app.selected_instance_preset_name {
                if let Some(selected) = app.selected_instance_preset_name.clone() {
                    app.apply_instance_preset_by_name(&selected);
                }
            }
            ui.label("Save as");
            ui.add_sized(
                [adaptive_field_width(ui, 0.34, 200.0, 320.0), 24.0],
                egui::TextEdit::singleline(&mut app.instance_preset_name_edit)
                    .hint_text("Preset name"),
            );
            if secondary_button_enabled(ui, "Save preset", app.selected_model_file_path.is_some())
                .clicked()
            {
                app.save_current_instance_preset();
            }
            if warning_button(ui, "Delete preset").clicked() {
                app.delete_selected_instance_preset();
            }
        });

        ui.add_space(8.0);
        ui.horizontal_wrapped(|ui| {
            ui.label("Instance name");
            let response = ui.add_sized(
                [adaptive_field_width(ui, 0.34, 200.0, 340.0), 24.0],
                egui::TextEdit::singleline(&mut app.create_params.name),
            );
            if response.changed() {
                app.sync_instance_name_edit_state();
            }
            ui.label("Retention");
            egui::ComboBox::from_id_salt("retention-mode")
                .selected_text(retention_label(app.create_params.retention_mode))
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut app.create_params.retention_mode,
                        RetentionMode::KeepLoaded,
                        "keep loaded",
                    );
                    ui.selectable_value(
                        &mut app.create_params.retention_mode,
                        RetentionMode::LoadOnDemand,
                        "load on demand",
                    );
                });
            render_load_on_demand_grace_editor(ui, app);
            ui.label("Max predict");
            ui.add(
                egui::DragValue::new(&mut app.chat_request.n_predict)
                    .range(-1..=32768)
                    .speed(16),
            );
        });

        ui.add_space(8.0);
        ui.horizontal_wrapped(|ui| {
            ui.label("Placement target");
            let mut target_choice = if auto_selected {
                "__auto__".to_string()
            } else {
                selected_target
                    .map(placement_target_choice_key)
                    .unwrap_or_else(|| "__auto__".to_string())
            };
            let previous_choice = target_choice.clone();
            egui::ComboBox::from_id_salt("integrated-placement-target")
                .selected_text(if auto_selected {
                    auto_target_label.to_string()
                } else {
                    selected_target
                        .map(|target| target.title.clone())
                        .unwrap_or_else(|| auto_target_label.to_string())
                })
                .width(adaptive_combo_width(ui, 0.72, 280.0, 680.0))
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut target_choice,
                        "__auto__".to_string(),
                        auto_target_label,
                    );
                    for target in targets {
                        ui.selectable_value(
                            &mut target_choice,
                            placement_target_choice_key(target),
                            format!("{} | {}", target.title, target.subtitle),
                        );
                    }
                });
            if secondary_button(ui, "Refresh targets").clicked() {
                let _ = app.refresh_cluster();
                let _ = app.refresh_telemetry();
                app.refresh_placement_candidates();
            }
            if previous_choice != target_choice {
                if target_choice == "__auto__" {
                    set_auto_placement_target(app);
                } else if let Some(target) = targets
                    .iter()
                    .find(|target| placement_target_choice_key(target) == target_choice)
                {
                    apply_placement_target(app, target);
                }
                app.last_plan = None;
                app.refresh_placement_candidates();
            }
        });

        if let Some(target) = selected_target {
            ui.horizontal_wrapped(|ui| {
                if auto_selected {
                    state_badge(
                        ui,
                        "best placement now",
                        egui::Color32::from_rgb(220, 252, 231),
                        egui::Color32::from_rgb(22, 101, 52),
                    );
                } else {
                    state_badge(
                        ui,
                        "pinned",
                        egui::Color32::from_rgb(224, 242, 254),
                        egui::Color32::from_rgb(14, 116, 144),
                    );
                }
                state_badge(
                    ui,
                    if !target.rpc_servers.trim().is_empty() {
                        "multi-node"
                    } else if target.execution_group_id.contains("split") {
                        "same-host split"
                    } else {
                        "single GPU"
                    },
                    if !target.rpc_servers.trim().is_empty() {
                        egui::Color32::from_rgb(255, 237, 213)
                    } else if target.execution_group_id.contains("split") {
                        egui::Color32::from_rgb(243, 232, 255)
                    } else {
                        egui::Color32::from_rgb(254, 249, 195)
                    },
                    if !target.rpc_servers.trim().is_empty() {
                        egui::Color32::from_rgb(154, 52, 18)
                    } else if target.execution_group_id.contains("split") {
                        egui::Color32::from_rgb(107, 33, 168)
                    } else {
                        egui::Color32::from_rgb(133, 77, 14)
                    },
                );
                state_badge(
                    ui,
                    if target.ready_now {
                        "ready now"
                    } else if target.requires_eviction {
                        "eviction needed"
                    } else {
                        "insufficient free memory"
                    },
                    if target.ready_now {
                        egui::Color32::from_rgb(220, 252, 231)
                    } else if target.requires_eviction {
                        egui::Color32::from_rgb(255, 237, 213)
                    } else {
                        egui::Color32::from_rgb(254, 226, 226)
                    },
                    if target.ready_now {
                        egui::Color32::from_rgb(22, 101, 52)
                    } else if target.requires_eviction {
                        egui::Color32::from_rgb(154, 52, 18)
                    } else {
                        egui::Color32::from_rgb(153, 27, 27)
                    },
                );
            });
            muted_label(ui, &target.subtitle);
        } else {
            muted_label(
                ui,
                "No eligible GPU targets are visible yet. Check node pairing, runtime install, and local model availability on the owner node.",
            );
        }

        let mut runtime_changed = false;
        ui.add_space(8.0);
        ui.horizontal_wrapped(|ui| {
            ui.label("n_ctx");
            runtime_changed |= ui
                .add(
                    egui::DragValue::new(&mut app.create_params.n_ctx)
                        .range(256..=262144)
                        .speed(256),
                )
                .changed();
            ui.label("n_batch");
            runtime_changed |= ui
                .add(
                    egui::DragValue::new(&mut app.create_params.n_batch)
                        .range(1..=32768)
                        .speed(32),
                )
                .changed();
            ui.label("n_ubatch");
            runtime_changed |= ui
                .add(
                    egui::DragValue::new(&mut app.create_params.n_ubatch)
                        .range(1..=32768)
                        .speed(32),
                )
                .changed();
            ui.label("gpu_layers");
            let mut gpu_layers_drag =
                egui::DragValue::new(&mut app.create_params.n_gpu_layers).speed(1);
            if let Some(layer_count) = detected_metadata.and_then(|metadata| metadata.block_count) {
                gpu_layers_drag = gpu_layers_drag.range(-1..=layer_count as i32);
            }
            runtime_changed |= ui.add(gpu_layers_drag).changed();
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("Parallel slots");
            for preset in [1, 2, 3] {
                let selected = app.create_params.n_parallel == preset;
                if ui
                    .selectable_label(selected, format!("{preset}"))
                    .on_hover_text(match preset {
                        1 => "Lowest VRAM pressure. Best for the largest runtimes.",
                        2 => "Keep one extra request slot warm if VRAM allows it.",
                        _ => "Reserve room for up to three active request contexts.",
                    })
                    .clicked()
                {
                    app.create_params.n_parallel = preset;
                    runtime_changed = true;
                }
            }
            runtime_changed |= ui
                .add(
                    egui::DragValue::new(&mut app.create_params.n_parallel)
                        .range(1..=8)
                        .speed(1),
                )
                .changed();
            if app.show_cpu_devices {
                ui.label("threads");
                runtime_changed |= ui
                    .add(egui::DragValue::new(&mut app.create_params.n_threads).speed(1))
                    .changed();
                ui.label("threads_batch");
                runtime_changed |= ui
                    .add(egui::DragValue::new(&mut app.create_params.n_threads_batch).speed(1))
                    .changed();
            }
        });

        if runtime_changed {
            app.last_plan = None;
            app.refresh_placement_candidates();
        }

        ui.add_space(8.0);
        outlined_card(ui, |ui| {
            ui.label(egui::RichText::new("Model insights").strong());
            if let Some(metadata) = detected_metadata {
                let mut summary = vec![metadata.format.clone()];
                if let Some(arch) = &metadata.architecture {
                    summary.push(arch.clone());
                }
                if let Some(layers) = metadata.block_count {
                    summary.push(format!("{layers} layers"));
                }
                if let Some(ctx) = metadata.trained_context_length {
                    summary.push(format!("trained ctx {ctx}"));
                }
                ui.label(summary.join(" | "));
            } else {
                ui.label("Layer count could not be detected automatically from this file.");
            }

            if let Some(estimate) = runtime_estimate {
                ui.label(format!(
                    "Approx GPU requirement: {}",
                    format_mib_from_bytes(estimate.required_gpu_bytes)
                ));
                ui.label(format!(
                    "Model {} | MMProj {} | KV cache {} | workspace {} | overhead {}",
                    format_mib_from_bytes(estimate.model_gpu_bytes),
                    format_mib_from_bytes(estimate.mmproj_gpu_bytes),
                    format_mib_from_bytes(estimate.kv_cache_bytes),
                    format_mib_from_bytes(estimate.workspace_bytes),
                    format_mib_from_bytes(estimate.overhead_bytes),
                ));
                if let (Some(total_layers), Some(on_gpu)) =
                    (estimate.detected_layer_count, estimate.layers_on_gpu)
                {
                    ui.label(format!(
                        "GPU offload: {} of {} layers",
                        on_gpu, total_layers
                    ));
                }
            } else {
                ui.label(
                    "Approx GPU requirement will appear after a primary model file is selected.",
                );
            }

            if let Some(metadata) = detected_metadata {
                if let Some(ctx) = metadata.trained_context_length {
                    if app.create_params.n_ctx > ctx as i32 {
                        muted_label(
                            ui,
                            "Current n_ctx is above the model metadata context. The estimate scales KV cache for it, but the backend may still clamp or reject it depending on the architecture.",
                        );
                    }
                }
            }

            if model.mmproj_path.is_some() {
                muted_label(
                    ui,
                    "MMProj stays whole on the selected primary GPU/device. The estimate follows the real bridge path here instead of pretending the projector is evenly split across GPUs.",
                );
            }
            muted_label(
                ui,
                "The backend currently supports one global gpu_layers value per load. Per-device manual layer maps are not exposed yet, so the split preview below is an estimate rather than a hard placement override.",
            );
        });

        if let (Some(target), Some(estimate)) = (selected_target, runtime_estimate) {
            ui.add_space(8.0);
            outlined_card(ui, |ui| {
                ui.label(egui::RichText::new("Approx target split").strong());
                render_target_distribution_preview(app, ui, target, estimate);
            });
        }

        ui.add_space(8.0);
        ui.horizontal_wrapped(|ui| {
            if accent_button(ui, "Plan placement").clicked() {
                app.plan_instance_cluster();
            }
            if secondary_button_enabled(
                ui,
                "Load now",
                selected_target
                    .map(placement_target_can_load)
                    .unwrap_or(false),
            )
            .clicked()
            {
                app.schedule_instance_cluster();
                open_instances_loaded_view(app);
            }
        });
    });

    if let Some(plan) = &app.last_plan {
        ui.add_space(8.0);
        outlined_card(ui, |ui| {
            ui.label(egui::RichText::new("Placement preview").strong().size(15.0));
            let placement_label = if plan.display_label.trim().is_empty() {
                plan.execution_group_id.clone()
            } else {
                plan.display_label.clone()
            };
            ui.label(format!(
                "{} on {} via {}",
                placement_strategy_label(plan.strategy),
                plan.owner_display_name,
                placement_label
            ));
            ui.label(format!(
                "Required {} | free {} | ready {} | reuse {} | eviction {}",
                format_mib_from_bytes(plan.estimated_required_bytes),
                format_mib_from_bytes(plan.estimated_group_free_bytes),
                yes_no(plan.ready_now),
                plan.reusable_instance_id
                    .map(|value: i64| value.to_string())
                    .unwrap_or_else(|| "<none>".to_string()),
                yes_no(plan.requires_eviction),
            ));
            if !plan.rpc_servers.is_empty() {
                ui.label(format!("Remote workers: {}", plan.rpc_servers));
            }
        });
    }
}

fn render_integrated_model_details(
    app: &mut ClusterControllerApp,
    ui: &mut egui::Ui,
    model: &ManagedModelEntry,
    targets: &[PlacementTargetView],
    auto_selected: bool,
    auto_target_label: &str,
    selected_target: Option<&PlacementTargetView>,
    detected_metadata: Option<&ModelFileMetadata>,
    runtime_estimate: Option<&RuntimeVramEstimate>,
) {
    let _ = (auto_selected, auto_target_label, runtime_estimate);
    ensure_manual_allocation_seed(app, targets, selected_target);
    if model.single_device_only && app.create_params.manual_device_allocations.len() > 1 {
        app.create_params.manual_device_allocations.truncate(1);
    }

    card(ui, Some("Launch setup"), |ui| {
        ui.label("Primary GPU defines the owner node. Add more local or RPC devices below and allocate layers directly.");
        ui.add_space(8.0);

        let compact_controls = compact_control_layout(ui);
        if compact_controls {
            ui.horizontal_wrapped(|ui| {
                ui.label("Preset");
                let previous_preset = app.selected_instance_preset_name.clone();
                egui::ComboBox::from_id_salt("instance-preset-picker")
                    .selected_text(
                        app.selected_instance_preset_name
                            .clone()
                            .unwrap_or_else(|| "No preset".to_string()),
                    )
                    .width(adaptive_combo_width(ui, 0.72, 200.0, 420.0))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut app.selected_instance_preset_name,
                            None,
                            "No preset",
                        );
                        for preset in &app.instance_presets {
                            ui.selectable_value(
                                &mut app.selected_instance_preset_name,
                                Some(preset.name.clone()),
                                &preset.name,
                            );
                        }
                    });
                if previous_preset != app.selected_instance_preset_name {
                    if let Some(selected) = app.selected_instance_preset_name.clone() {
                        app.apply_instance_preset_by_name(&selected);
                    }
                }
            });
            ui.horizontal_wrapped(|ui| {
                ui.label("Save as");
                ui.add_sized(
                    [adaptive_field_width(ui, 0.72, 200.0, 420.0), 24.0],
                    egui::TextEdit::singleline(&mut app.instance_preset_name_edit)
                        .hint_text("Preset name"),
                );
            });
            ui.horizontal_wrapped(|ui| {
                if secondary_button_enabled(
                    ui,
                    "Save preset",
                    app.selected_model_file_path.is_some(),
                )
                .clicked()
                {
                    app.save_current_instance_preset();
                }
                if warning_button(ui, "Delete preset").clicked() {
                    app.delete_selected_instance_preset();
                }
            });
        } else {
            ui.horizontal_wrapped(|ui| {
                ui.label("Preset");
                let previous_preset = app.selected_instance_preset_name.clone();
                egui::ComboBox::from_id_salt("instance-preset-picker")
                    .selected_text(
                        app.selected_instance_preset_name
                            .clone()
                            .unwrap_or_else(|| "No preset".to_string()),
                    )
                    .width(adaptive_combo_width(ui, 0.34, 200.0, 320.0))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut app.selected_instance_preset_name,
                            None,
                            "No preset",
                        );
                        for preset in &app.instance_presets {
                            ui.selectable_value(
                                &mut app.selected_instance_preset_name,
                                Some(preset.name.clone()),
                                &preset.name,
                            );
                        }
                    });
                if previous_preset != app.selected_instance_preset_name {
                    if let Some(selected) = app.selected_instance_preset_name.clone() {
                        app.apply_instance_preset_by_name(&selected);
                    }
                }
                ui.label("Save as");
                ui.add_sized(
                    [adaptive_field_width(ui, 0.34, 200.0, 320.0), 24.0],
                    egui::TextEdit::singleline(&mut app.instance_preset_name_edit)
                        .hint_text("Preset name"),
                );
                if secondary_button_enabled(
                    ui,
                    "Save preset",
                    app.selected_model_file_path.is_some(),
                )
                .clicked()
                {
                    app.save_current_instance_preset();
                }
                if warning_button(ui, "Delete preset").clicked() {
                    app.delete_selected_instance_preset();
                }
            });
        }

        ui.add_space(8.0);
        if compact_controls {
            ui.horizontal_wrapped(|ui| {
                ui.label("Instance name");
                let response = ui.add_sized(
                    [adaptive_field_width(ui, 0.72, 220.0, 460.0), 24.0],
                    egui::TextEdit::singleline(&mut app.create_params.name),
                );
                if response.changed() {
                    app.sync_instance_name_edit_state();
                }
            });
            ui.horizontal_wrapped(|ui| {
                ui.label("Retention");
                egui::ComboBox::from_id_salt("retention-mode")
                    .selected_text(retention_label(app.create_params.retention_mode))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut app.create_params.retention_mode,
                            RetentionMode::KeepLoaded,
                            "keep loaded",
                        );
                        ui.selectable_value(
                            &mut app.create_params.retention_mode,
                            RetentionMode::LoadOnDemand,
                            "load on demand",
                        );
                    });
                render_load_on_demand_grace_editor(ui, app);
            });
            ui.horizontal_wrapped(|ui| {
                ui.label("Max predict");
                ui.add(
                    egui::DragValue::new(&mut app.chat_request.n_predict)
                        .range(-1..=65536)
                        .speed(16),
                );
            });
        } else {
            ui.horizontal_wrapped(|ui| {
                ui.label("Instance name");
                let response = ui.add_sized(
                    [adaptive_field_width(ui, 0.34, 200.0, 340.0), 24.0],
                    egui::TextEdit::singleline(&mut app.create_params.name),
                );
                if response.changed() {
                    app.sync_instance_name_edit_state();
                }
                ui.label("Retention");
                egui::ComboBox::from_id_salt("retention-mode")
                    .selected_text(retention_label(app.create_params.retention_mode))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut app.create_params.retention_mode,
                            RetentionMode::KeepLoaded,
                            "keep loaded",
                        );
                        ui.selectable_value(
                            &mut app.create_params.retention_mode,
                            RetentionMode::LoadOnDemand,
                            "load on demand",
                        );
                    });
                render_load_on_demand_grace_editor(ui, app);
                ui.label("Max predict");
                ui.add(
                    egui::DragValue::new(&mut app.chat_request.n_predict)
                        .range(-1..=65536)
                        .speed(16),
                );
            });
        }

        let total_layers = detected_metadata.and_then(|metadata| metadata.block_count);
        let total_layers_field_text = total_layers
            .map(|value| value.to_string())
            .unwrap_or_else(|| "?".to_string());
        let owner_choices = manual_owner_choices(app);
        let mut manual_choices = manual_device_choices(app);
        if app.create_params.manual_device_allocations.is_empty() {
            if let Some(primary_choice) = manual_choices.iter().find(|choice| !choice.rpc_device) {
                app.create_params.manual_device_allocations.push(
                    crate::cluster_api::ManualDeviceAllocation {
                        bridge_device_index: primary_choice.bridge_device_index,
                        device_label: primary_choice.device_label.clone(),
                        backend: primary_choice.backend.clone(),
                        layer_count: 0,
                        rpc_device: false,
                        source_node_id: primary_choice.source_node_id.clone(),
                        source_control_addr: primary_choice.source_control_addr.clone(),
                        source_bridge_device_index: primary_choice.source_bridge_device_index,
                    },
                );
            }
        }

        let mut runtime_changed = false;
        ui.add_space(8.0);
        outlined_card(ui, |ui| {
            ui.label(egui::RichText::new("Device allocation").strong());
            if owner_choices.is_empty() {
                muted_label(
                    ui,
                    "No visible GPU devices are available yet. Connect nodes, start runtimes, and refresh the cluster state.",
                );
                if secondary_button(ui, "Refresh devices").clicked() {
                    let _ = app.refresh_cluster();
                    let _ = app.refresh_telemetry();
                    let _ = app.refresh_selected_preview();
                }
                return;
            }
            if app.create_params.manual_device_allocations.is_empty() {
                muted_label(ui, "Select a primary GPU to start assigning layers.");
                return;
            }
            let current_owner_key = app
                .manual_owner_control_addr()
                .zip(app.create_params.manual_device_allocations.first())
                .map(|(owner, row)| {
                    format!(
                        "{owner}|{}",
                        if row.source_bridge_device_index >= 0 {
                            row.source_bridge_device_index
                        } else {
                            row.bridge_device_index
                        }
                    )
                })
                .unwrap_or_else(|| manual_owner_choice_key(&owner_choices[0]));
            let mut next_owner_key = current_owner_key.clone();
            let current_owner_label = owner_choices
                .iter()
                .find(|choice| manual_owner_choice_key(choice) == current_owner_key)
                .map(|choice| {
                    format!(
                        "{} | {} [{}]",
                        choice.owner_display_name, choice.device_label, choice.backend
                    )
                })
                .unwrap_or_else(|| "Select a primary GPU".to_string());
            let owner_availability_badges = owner_choices
                .iter()
                .filter_map(|choice| {
                    owner_artifact_availability_badge(app, &choice.owner_control_addr)
                        .map(|badge| (choice.owner_control_addr.clone(), badge))
                })
                .collect::<BTreeMap<_, _>>();
            let mut current_owner_badge = app
                .manual_owner_control_addr()
                .and_then(|owner_control_addr| owner_availability_badges.get(owner_control_addr))
                .cloned();

            let mut refresh_devices_requested = false;
            let current_owner_action_label =
                app.manual_owner_control_addr()
                    .and_then(|owner_control_addr| {
                        owner_transfer_action_label(app, owner_control_addr)
                    });
            let mut start_owner_transfer = false;
            let primary_slider_max =
                manual_layer_slider_max(total_layers, &app.create_params.manual_device_allocations);
            let allow_primary_auto_full_offload =
                app.create_params.manual_device_allocations.len() == 1;
            if let Some(row) = app.create_params.manual_device_allocations.first_mut() {
                let live_choice = manual_choices
                    .iter()
                    .find(|choice| !choice.rpc_device && manual_allocation_matches_choice(row, choice))
                    .cloned();
                if let Some(choice) = &live_choice {
                    row.device_label = choice.device_label.clone();
                    row.backend = choice.backend.clone();
                    row.rpc_device = choice.rpc_device;
                    row.source_node_id = choice.source_node_id.clone();
                    row.source_control_addr = choice.source_control_addr.clone();
                    row.source_bridge_device_index = choice.source_bridge_device_index;
                }

                outlined_card(ui, |ui| {
                    ui.label(egui::RichText::new("Primary device").strong());
                    ui.horizontal_wrapped(|ui| {
                        egui::ComboBox::from_id_salt("manual-primary-device")
                            .selected_text(current_owner_label.clone())
                            .width(adaptive_combo_width(ui, 0.72, 280.0, 720.0))
                            .show_ui(ui, |ui| {
                                for choice in &owner_choices {
                                    let choice_key = manual_owner_choice_key(choice);
                                    let selected = next_owner_key == choice_key;
                                    if render_combo_choice_with_badge(
                                        ui,
                                        selected,
                                        format!(
                                            "{} | {} [{}] | {} free / {}",
                                            choice.owner_display_name,
                                            choice.device_label,
                                            choice.backend,
                                            format_mib(choice.memory_free),
                                            format_mib(choice.memory_total),
                                        ),
                                        owner_availability_badges.get(&choice.owner_control_addr),
                                    ) {
                                        next_owner_key = choice_key;
                                    }
                                }
                            });
                        if secondary_button(ui, "Refresh devices").clicked() {
                            refresh_devices_requested = true;
                        }
                    });
                    if let Some(badge) = current_owner_badge.as_ref() {
                        ui.horizontal_wrapped(|ui| {
                            state_badge(ui, &badge.label, badge.fill, badge.text);
                            if badge.label != "available" {
                                if let Some(action_label) = current_owner_action_label.as_ref() {
                                    if secondary_button_enabled(
                                        ui,
                                        action_label,
                                        !app.model_transfer_in_progress,
                                    )
                                    .clicked()
                                    {
                                        start_owner_transfer = true;
                                    }
                                }
                            }
                        });
                    }
                    let mut layer_value = if allow_primary_auto_full_offload && row.layer_count == 0
                    {
                        -1
                    } else {
                        i32::try_from(row.layer_count).unwrap_or(i32::MAX)
                    };
                    ui.horizontal_wrapped(|ui| {
                        ui.add(
                            egui::Slider::new(
                                &mut layer_value,
                                if allow_primary_auto_full_offload {
                                    -1..=primary_slider_max
                                } else {
                                    0..=primary_slider_max
                                },
                            )
                            .show_value(false)
                            .clamping(egui::SliderClamping::Always),
                        );
                        let layer_drag = if total_layers.is_some() {
                            if allow_primary_auto_full_offload {
                                egui::DragValue::new(&mut layer_value)
                                    .range(-1..=primary_slider_max)
                                    .speed(1)
                            } else {
                                egui::DragValue::new(&mut layer_value)
                                    .range(0..=primary_slider_max)
                                    .speed(1)
                            }
                        } else {
                            egui::DragValue::new(&mut layer_value).speed(1)
                        };
                        runtime_changed |= ui.add(layer_drag).changed();
                        ui.label("/");
                        let mut total_layers_value = total_layers_field_text.clone();
                        ui.add_enabled_ui(false, |ui| {
                            ui.add_sized(
                                [56.0, 24.0],
                                egui::TextEdit::singleline(&mut total_layers_value),
                            );
                        });
                        ui.label("layers");
                    });
                    let next_layer_count = if allow_primary_auto_full_offload && layer_value < 0 {
                        0
                    } else {
                        u32::try_from(layer_value.max(0)).unwrap_or_default()
                    };
                    if next_layer_count != row.layer_count {
                        row.layer_count = next_layer_count;
                        runtime_changed = true;
                    }
                });
            }
            if start_owner_transfer {
                app.start_selected_owner_artifact_transfer();
            }
            if refresh_devices_requested {
                let _ = app.refresh_cluster();
                let _ = app.refresh_telemetry();
                let _ = app.refresh_selected_preview();
            }
            if next_owner_key != current_owner_key {
                if let Some(choice) = owner_choices
                    .iter()
                    .find(|choice| manual_owner_choice_key(choice) == next_owner_key)
                    .cloned()
                {
                    set_manual_primary_owner(app, &choice);
                    app.selected_rpc_peer_addrs = app
                        .all_visible_rpc_peer_control_addrs_for_owner(
                            &choice.owner_control_addr,
                        );
                    let _ = app.refresh_selected_preview();
                    runtime_changed = true;
                    manual_choices = manual_device_choices(app);
                    current_owner_badge = app
                        .manual_owner_control_addr()
                        .and_then(|owner_control_addr| {
                            owner_availability_badges.get(owner_control_addr)
                        })
                        .cloned();
                }
            }

            let current_choice_keys = app
                .create_params
                .manual_device_allocations
                .iter()
                .filter_map(manual_allocation_choice_key)
                .collect::<Vec<_>>();
            for row_index in 1..app.create_params.manual_device_allocations.len() {
                let row_slider_max = manual_layer_slider_max(
                    total_layers,
                    &app.create_params.manual_device_allocations,
                );
                let row = &mut app.create_params.manual_device_allocations[row_index];
                let live_choice = manual_choices
                    .iter()
                    .find(|choice| manual_allocation_matches_choice(row, choice))
                    .cloned();
                if let Some(choice) = &live_choice {
                    row.device_label = choice.device_label.clone();
                    row.backend = choice.backend.clone();
                    row.rpc_device = choice.rpc_device;
                    row.source_node_id = choice.source_node_id.clone();
                    row.source_control_addr = choice.source_control_addr.clone();
                    row.source_bridge_device_index = choice.source_bridge_device_index;
                }

                outlined_card(ui, |ui| {
                    ui.label(egui::RichText::new("Additional device").strong());
                    let available_choices = manual_choices
                        .iter()
                        .filter(|choice| {
                            let choice_key = manual_device_choice_key(choice);
                            !current_choice_keys.iter().enumerate().any(|(index, used)| {
                                index != row_index && *used == choice_key
                            })
                        })
                        .cloned()
                        .collect::<Vec<_>>();
                    let mut current_choice_key = live_choice
                        .as_ref()
                        .map(manual_device_choice_key)
                        .or_else(|| manual_allocation_choice_key(row))
                        .unwrap_or_else(|| format!("{}|{}", row.bridge_device_index, row.rpc_device));
                    egui::ComboBox::from_id_salt(format!("manual-device-row-{row_index}"))
                        .selected_text(
                            live_choice
                                .as_ref()
                                .map(|choice| {
                                    format!(
                                        "{} [{}]",
                                        manual_device_choice_display_label(choice),
                                        choice.backend
                                    )
                                })
                                .unwrap_or_else(|| format!("{} [{}]", row.device_label, row.backend)),
                        )
                        .width(adaptive_combo_width(ui, 0.68, 280.0, 700.0))
                        .show_ui(ui, |ui| {
                            for choice in &available_choices {
                                let choice_key = manual_device_choice_key(choice);
                                let selected = current_choice_key == choice_key;
                                if render_combo_choice_with_badge(
                                    ui,
                                    selected,
                                    format!(
                                        "{} [{}] | {} free / {}{}",
                                        manual_device_choice_display_label(choice),
                                        choice.backend,
                                        format_mib(choice.memory_free),
                                        format_mib(choice.memory_total),
                                        if choice.rpc_device { " | RPC" } else { "" }
                                    ),
                                    current_owner_badge.as_ref(),
                                ) {
                                    current_choice_key = choice_key;
                                }
                            }
                        });
                    if let Some(choice) = available_choices
                        .iter()
                        .find(|choice| manual_device_choice_key(choice) == current_choice_key)
                    {
                        if !manual_allocation_matches_choice(row, choice) {
                            apply_manual_allocation_row_from_choice(row, choice);
                            runtime_changed = true;
                        }
                    }

                    let mut layer_value = i32::try_from(row.layer_count).unwrap_or(i32::MAX);
                    ui.horizontal_wrapped(|ui| {
                        ui.add(
                            egui::Slider::new(&mut layer_value, 0..=row_slider_max)
                                .show_value(false)
                                .clamping(egui::SliderClamping::Always),
                        );
                        let layer_drag = if total_layers.is_some() {
                            egui::DragValue::new(&mut layer_value)
                                .range(0..=row_slider_max)
                                .speed(1)
                        } else {
                            egui::DragValue::new(&mut layer_value).speed(1)
                        };
                        runtime_changed |= ui.add(layer_drag).changed();
                        ui.label("/");
                        let mut total_layers_value = total_layers_field_text.clone();
                        ui.add_enabled_ui(false, |ui| {
                            ui.add_sized(
                                [56.0, 24.0],
                                egui::TextEdit::singleline(&mut total_layers_value),
                            );
                        });
                        ui.label("layers");
                        if warning_button(ui, "Remove").clicked() {
                            row.bridge_device_index = -1;
                        }
                    });
                    let next_layer_count = u32::try_from(layer_value.max(0)).unwrap_or_default();
                    if next_layer_count != row.layer_count {
                        row.layer_count = next_layer_count;
                        runtime_changed = true;
                    }
                });
            }

            let before_len = app.create_params.manual_device_allocations.len();
            app.create_params
                .manual_device_allocations
                .retain(|row| row.bridge_device_index >= 0);
            if app.create_params.manual_device_allocations.len() != before_len {
                runtime_changed = true;
            }

            if !model.single_device_only {
                let used_choice_keys = app
                    .create_params
                    .manual_device_allocations
                    .iter()
                    .filter_map(manual_allocation_choice_key)
                    .collect::<BTreeSet<_>>();
                let next_choice = manual_choices
                    .iter()
                    .find(|choice| !used_choice_keys.contains(&manual_device_choice_key(choice)))
                    .cloned();
                ui.add_space(6.0);
                if secondary_button_enabled(ui, "+ Add device", next_choice.is_some()).clicked() {
                    if let Some(choice) = next_choice {
                        app.create_params.manual_device_allocations.push(
                            crate::cluster_api::ManualDeviceAllocation {
                                bridge_device_index: choice.bridge_device_index,
                                device_label: choice.device_label.clone(),
                                backend: choice.backend.clone(),
                                layer_count: 0,
                                rpc_device: choice.rpc_device,
                                source_node_id: choice.source_node_id.clone(),
                                source_control_addr: choice.source_control_addr.clone(),
                                source_bridge_device_index: choice.source_bridge_device_index,
                            },
                        );
                        runtime_changed = true;
                    }
                }
            }
        });

        app.create_params.n_gpu_layers = app.manual_selected_gpu_layers(total_layers);
        app.create_params.manual_devices_csv = None;
        app.create_params.manual_tensor_split = None;

        ui.add_space(8.0);
        if compact_controls {
            ui.horizontal_wrapped(|ui| {
                ui.label("n_ctx");
                runtime_changed |= ui
                    .add(
                        egui::DragValue::new(&mut app.create_params.n_ctx)
                            .range(256..=262144)
                            .speed(256),
                    )
                    .changed();
                ui.label("n_batch");
                runtime_changed |= ui
                    .add(
                        egui::DragValue::new(&mut app.create_params.n_batch)
                            .range(1024..=32768)
                            .speed(32),
                    )
                    .changed();
            });
            ui.horizontal_wrapped(|ui| {
                ui.label("n_ubatch");
                runtime_changed |= ui
                    .add(
                        egui::DragValue::new(&mut app.create_params.n_ubatch)
                            .range(1024..=32768)
                            .speed(32),
                    )
                    .changed();
                ui.label("GPU layers");
                ui.monospace(app.create_params.n_gpu_layers.to_string());
                ui.label("/");
                let mut total_layers_value = total_layers_field_text.clone();
                ui.add_enabled_ui(false, |ui| {
                    ui.add_sized(
                        [56.0, 24.0],
                        egui::TextEdit::singleline(&mut total_layers_value),
                    );
                });
                ui.label("total");
            });
        } else {
            ui.horizontal_wrapped(|ui| {
                ui.label("n_ctx");
                runtime_changed |= ui
                    .add(
                        egui::DragValue::new(&mut app.create_params.n_ctx)
                            .range(256..=262144)
                            .speed(256),
                    )
                    .changed();
                ui.label("n_batch");
                runtime_changed |= ui
                    .add(
                        egui::DragValue::new(&mut app.create_params.n_batch)
                            .range(1024..=32768)
                            .speed(32),
                    )
                    .changed();
                ui.label("n_ubatch");
                runtime_changed |= ui
                    .add(
                        egui::DragValue::new(&mut app.create_params.n_ubatch)
                            .range(1024..=32768)
                            .speed(32),
                    )
                    .changed();
                ui.label("GPU layers");
                ui.monospace(app.create_params.n_gpu_layers.to_string());
                ui.label("/");
                let mut total_layers_value = total_layers_field_text.clone();
                ui.add_enabled_ui(false, |ui| {
                    ui.add_sized(
                        [56.0, 24.0],
                        egui::TextEdit::singleline(&mut total_layers_value),
                    );
                });
                ui.label("total");
            });
        }
        ui.horizontal_wrapped(|ui| {
            ui.label("Parallel slots");
            for preset in [1, 2, 3] {
                let selected = app.create_params.n_parallel == preset;
                if ui
                    .selectable_label(selected, format!("{preset}"))
                    .on_hover_text(match preset {
                        1 => "Lowest VRAM pressure. Best for the largest runtimes.",
                        2 => "Keep one extra request slot warm if VRAM allows it.",
                        _ => "Reserve room for up to three active request contexts.",
                    })
                    .clicked()
                {
                    app.create_params.n_parallel = preset;
                    runtime_changed = true;
                }
            }
            runtime_changed |= ui
                .add(
                    egui::DragValue::new(&mut app.create_params.n_parallel)
                        .range(1..=8)
                        .speed(1),
                )
                .changed();
            muted_label(
                ui,
                "Cluster launch now auto-uses all visible CPU threads on the owner for tokenization and host-side work.",
            );
        });

        if runtime_changed {
            app.last_plan = None;
        }

        let runtime_estimate = app.selected_runtime_vram_estimate();
        let allocation_summary =
            manual_allocation_summary(app, runtime_estimate.as_ref(), total_layers);
        let total_layers_text = allocation_summary
            .total_layers
            .map(|value| value.to_string())
            .unwrap_or_else(|| "?".to_string());

        ui.add_space(8.0);
        ui.horizontal_wrapped(|ui| {
            state_badge(
                ui,
                "pinned",
                egui::Color32::from_rgb(224, 242, 254),
                egui::Color32::from_rgb(14, 116, 144),
            );
            state_badge(
                ui,
                allocation_summary.shape_label,
                if allocation_summary.shape_label == "multi-node split" {
                    egui::Color32::from_rgb(255, 237, 213)
                } else if allocation_summary.shape_label == "same-host split" {
                    egui::Color32::from_rgb(243, 232, 255)
                } else {
                    egui::Color32::from_rgb(254, 249, 195)
                },
                if allocation_summary.shape_label == "multi-node split" {
                    egui::Color32::from_rgb(154, 52, 18)
                } else if allocation_summary.shape_label == "same-host split" {
                    egui::Color32::from_rgb(107, 33, 168)
                } else {
                    egui::Color32::from_rgb(133, 77, 14)
                },
            );
            if let Some(fit) = allocation_summary.fit {
                let (label, fill, text) = match fit {
                    ManualAllocationFit::ReadyNow => (
                        "ready now",
                        egui::Color32::from_rgb(220, 252, 231),
                        egui::Color32::from_rgb(22, 101, 52),
                    ),
                    ManualAllocationFit::EvictionNeeded => (
                        "eviction needed",
                        egui::Color32::from_rgb(255, 237, 213),
                        egui::Color32::from_rgb(154, 52, 18),
                    ),
                    ManualAllocationFit::Insufficient => (
                        "insufficient visible memory",
                        egui::Color32::from_rgb(254, 226, 226),
                        egui::Color32::from_rgb(153, 27, 27),
                    ),
                };
                state_badge(ui, label, fill, text);
            }
            state_badge(
                ui,
                &format!(
                    "allocated {} / {} layers",
                    if manual_single_device_full_offload(app) {
                        "-1".to_string()
                    } else {
                        allocation_summary.allocated_layers.to_string()
                    },
                    total_layers_text
                ),
                egui::Color32::from_rgb(239, 246, 255),
                egui::Color32::from_rgb(30, 64, 175),
            );
        });

        outlined_card(ui, |ui| {
            ui.label(egui::RichText::new("Model insights").strong());
            if let Some(metadata) = detected_metadata {
                let mut summary = vec![metadata.format.clone()];
                if let Some(arch) = &metadata.architecture {
                    summary.push(arch.clone());
                }
                if let Some(layers) = metadata.block_count {
                    summary.push(format!("{layers} layers"));
                }
                if let Some(ctx) = metadata.trained_context_length {
                    summary.push(format!("trained ctx {ctx}"));
                }
                ui.label(summary.join(" | "));
            } else {
                ui.label("Layer count could not be detected automatically from this file.");
            }

            if let Some(estimate) = runtime_estimate.as_ref() {
                ui.label(format!(
                    "Approx GPU requirement: {}",
                    format_mib_from_bytes(estimate.required_gpu_bytes)
                ));
                ui.label(format!(
                    "Model {} | MMProj {} | KV cache {} | workspace {} | overhead {}",
                    format_mib_from_bytes(estimate.model_gpu_bytes),
                    format_mib_from_bytes(estimate.mmproj_gpu_bytes),
                    format_mib_from_bytes(estimate.kv_cache_bytes),
                    format_mib_from_bytes(estimate.workspace_bytes),
                    format_mib_from_bytes(estimate.overhead_bytes),
                ));
            } else {
                ui.label(
                    "Approx GPU requirement will appear after a primary model file is selected.",
                );
            }
            muted_label(
                ui,
                "Launch is not blocked by fit math. These numbers stay advisory because manual allocations can still work beyond the estimator.",
            );
        });

        if !allocation_summary.row_estimates.is_empty() {
            ui.add_space(8.0);
            outlined_card(ui, |ui| {
                ui.label(egui::RichText::new("Per-device estimate").strong());
                for (index, row) in allocation_summary.row_estimates.iter().enumerate() {
                    ui.label(
                        egui::RichText::new(format!(
                            "{} [{}]{}",
                            row.device_label,
                            row.backend,
                            if row.rpc_device { " | RPC" } else { "" }
                        ))
                        .strong(),
                    );
                    let ratio = if row.memory_total == 0 {
                        0.0
                    } else {
                        memory_ratio(row.estimated_bytes, row.memory_total)
                    };
                    ui.add(
                        egui::ProgressBar::new(ratio)
                            .desired_width(ui.available_width())
                            .text(if row.available {
                                format!(
                                    "{} est. of {}",
                                    format_mib(row.estimated_bytes),
                                    format_mib(row.memory_total)
                                )
                            } else {
                                "device not visible".to_string()
                            }),
                    );
                    if row.available {
                        ui.label(format!(
                            "{} layers | {} free right now",
                            if manual_single_device_full_offload(app) && index == 0 {
                                "-1".to_string()
                            } else {
                                row.layer_count.to_string()
                            },
                            format_mib(row.memory_free)
                        ));
                    } else {
                        ui.label(format!(
                            "{} layers",
                            if manual_single_device_full_offload(app) && index == 0 {
                                "-1".to_string()
                            } else {
                                row.layer_count.to_string()
                            }
                        ));
                    }
                    if index == 0 && model.mmproj_path.is_some() {
                        muted_label(ui, "Includes the full MMProj on the primary GPU.");
                    }
                    ui.add_space(6.0);
                }
            });
        }

        ui.add_space(8.0);
        ui.horizontal_wrapped(|ui| {
            if secondary_button(ui, "Refresh cluster state").clicked() {
                app.refresh_all_ui();
            }
            ui.add_enabled_ui(
                !app.create_params.manual_device_allocations.is_empty()
                    && !app.manual_load_in_progress,
                |ui| {
                    if accent_button(ui, "Load now").clicked() {
                        app.load_manual_instance_cluster();
                    }
                },
            );
            if app.manual_load_in_progress {
                ui.spinner();
                muted_label(ui, "Loading cluster runtime...");
            }
        });
    });
}

#[derive(Clone)]
struct PlacementTargetView {
    owner_control_addr: String,
    owner_display_name: String,
    execution_group_id: String,
    rpc_servers: String,
    title: String,
    subtitle: String,
    estimated_required_bytes: u64,
    estimated_group_free_bytes: u64,
    ready_now: bool,
    requires_eviction: bool,
}

#[derive(Clone)]
struct ManualOwnerChoice {
    owner_control_addr: String,
    owner_display_name: String,
    bridge_device_index: i32,
    device_label: String,
    backend: String,
    memory_free: u64,
    memory_total: u64,
}

#[derive(Clone)]
struct ManualDeviceChoice {
    source_node_id: String,
    source_control_addr: String,
    source_display_name: String,
    source_bridge_device_index: i32,
    bridge_device_index: i32,
    device_label: String,
    backend: String,
    memory_free: u64,
    memory_total: u64,
    rpc_device: bool,
}

#[derive(Clone)]
struct OwnerArtifactAvailabilityBadge {
    label: String,
    fill: egui::Color32,
    text: egui::Color32,
}

#[derive(Clone)]
struct ManualAllocationEstimate {
    device_label: String,
    backend: String,
    layer_count: u32,
    estimated_bytes: u64,
    memory_free: u64,
    memory_total: u64,
    rpc_device: bool,
    available: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ManualAllocationFit {
    ReadyNow,
    EvictionNeeded,
    Insufficient,
}

struct ManualAllocationSummary {
    allocated_layers: u32,
    total_layers: Option<u32>,
    shape_label: &'static str,
    fit: Option<ManualAllocationFit>,
    row_estimates: Vec<ManualAllocationEstimate>,
}

fn is_cpu_device(device: &DeviceInfo) -> bool {
    let lowered = format!("{} {}", device.backend, device.name).to_ascii_lowercase();
    lowered.contains("cpu") || lowered.contains("blas") || lowered.contains("accelerate")
}

fn is_rpc_device(device: &DeviceInfo) -> bool {
    let lowered = format!("{} {}", device.backend, device.name).to_ascii_lowercase();
    lowered.contains("rpc")
}

fn is_metal_device(device: &DeviceInfo) -> bool {
    let lowered = format!("{} {}", device.backend, device.name).to_ascii_lowercase();
    lowered.contains("metal")
}

fn device_is_integrated_gpu(node: &NodeSnapshot, device: &DeviceInfo) -> bool {
    if is_cpu_device(device) || is_rpc_device(device) {
        return false;
    }
    if node.node.os_name.eq_ignore_ascii_case("macos") && is_metal_device(device) {
        return false;
    }
    let lowered = format!("{} {}", device.name, device.description).to_ascii_lowercase();
    let looks_intel_integrated = lowered.contains("intel") && !lowered.contains("arc");
    let looks_integrated_family = lowered.contains("integrated")
        || lowered.contains("uhd")
        || lowered.contains("iris")
        || lowered.contains("hd graphics")
        || lowered.contains("xe graphics")
        || lowered.contains("graphics controller")
        || lowered.contains("apu")
        || lowered.contains("uma");
    let looks_shared_memory = lowered.contains("shared")
        || lowered.contains("unified")
        || lowered.contains("system memory");
    looks_intel_integrated || looks_integrated_family || looks_shared_memory
}

fn device_visible_in_ui(
    app: &ClusterControllerApp,
    node: &NodeSnapshot,
    device: &DeviceInfo,
) -> bool {
    if is_cpu_device(device) {
        return app.show_cpu_devices;
    }
    if !app.show_integrated_gpus && device_is_integrated_gpu(node, device) {
        return false;
    }
    true
}

fn filtered_devices_for_node(
    app: &ClusterControllerApp,
    node: &NodeSnapshot,
    telemetry: Option<&TelemetrySnapshot>,
) -> Vec<DeviceInfo> {
    let devices = raw_visible_devices_for_node(app, node, telemetry);
    dedupe_visible_devices(node, devices)
}

fn raw_visible_devices_for_node(
    app: &ClusterControllerApp,
    node: &NodeSnapshot,
    telemetry: Option<&TelemetrySnapshot>,
) -> Vec<DeviceInfo> {
    telemetry
        .map(|snapshot| snapshot.devices.clone())
        .unwrap_or_else(|| node.devices.clone())
        .into_iter()
        .filter(|device| device_visible_in_ui(app, node, device))
        .collect()
}

fn device_display_name(_node: &NodeSnapshot, device: &DeviceInfo) -> String {
    let description = device.description.trim();
    if !description.is_empty() && !description.eq_ignore_ascii_case(device.name.trim()) {
        description.to_string()
    } else {
        device.name.clone()
    }
}

fn device_source_display_name(node: &NodeSnapshot) -> String {
    node.node.display_name.clone()
}

fn device_display_name_for_ui(
    app: &ClusterControllerApp,
    node: &NodeSnapshot,
    device: &DeviceInfo,
) -> String {
    if !is_rpc_device(device) {
        return device_display_name(node, device);
    }
    let Some(endpoint) = rpc_endpoint_from_device(device) else {
        return device_display_name(node, device);
    };
    let Some(remote_node) = lookup_node_for_rpc_endpoint(app, &endpoint) else {
        return device_display_name(node, device);
    };

    let remote_devices = filtered_devices_for_node(
        app,
        remote_node,
        app.telemetry_for_control_addr(&remote_node.control_addr),
    )
    .into_iter()
    .filter(|candidate| !is_cpu_device(candidate) && !is_rpc_device(candidate))
    .collect::<Vec<_>>();
    if remote_devices.is_empty() {
        return remote_node.node.display_name.clone();
    }

    if let Some(exact) = unique_remote_device_match_by_memory(&remote_devices, device.memory_total) {
        return format!(
            "{} | {}",
            remote_node.node.display_name,
            device_display_name(remote_node, exact)
        );
    }

    let rpc_ordinal = rpc_device_ordinal_for_endpoint(node, device, &endpoint);
    let mut sorted_remote_devices = remote_devices;
    sorted_remote_devices.sort_by(|lhs, rhs| {
        lhs.bridge_device_index
            .cmp(&rhs.bridge_device_index)
            .then(lhs.memory_total.cmp(&rhs.memory_total))
            .then(lhs.name.cmp(&rhs.name))
    });
    if let Some(matched) = sorted_remote_devices.get(rpc_ordinal) {
        return format!(
            "{} | {}",
            remote_node.node.display_name,
            device_display_name(remote_node, matched)
        );
    }

    format!(
        "{} | Remote GPU ({})",
        remote_node.node.display_name,
        format_mib(device.memory_total)
    )
}

fn rpc_endpoint_from_device(device: &DeviceInfo) -> Option<String> {
    normalize_rpc_endpoint_token(device.description.trim())
        .or_else(|| normalize_rpc_endpoint_token(device.name.trim()))
}

fn normalize_rpc_endpoint_token(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    let trimmed = trimmed
        .strip_prefix("rpc://")
        .unwrap_or(trimmed)
        .trim_end_matches(" [RPC]")
        .trim();
    trimmed.contains(':').then(|| trimmed.to_string())
}

fn rpc_endpoint_candidates_for_node(node: &NodeSnapshot) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    let mut push = |value: String| {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return;
        }
        if seen.insert(trimmed.to_string()) {
            out.push(trimmed.to_string());
        }
    };

    if let Some(value) = node.advertised_rpc_endpoint.as_deref() {
        push(value.to_string());
    }
    if let Some(value) = node.rpc_endpoint.as_deref() {
        push(value.to_string());
    }

    let rpc_port = node
        .advertised_rpc_endpoint
        .as_deref()
        .or(node.rpc_endpoint.as_deref())
        .and_then(|value| value.rsplit_once(':').map(|(_, port)| port.to_string()))
        .unwrap_or_else(|| CLUSTER_AGENT_RPC_PORT.to_string());
    for control_addr in display_control_paths(&node.control_addr, &node.known_control_addrs) {
        if let Some((host, _)) = control_addr.rsplit_once(':') {
            push(format!("{host}:{rpc_port}"));
        }
    }
    out
}

fn lookup_node_for_rpc_endpoint<'a>(
    app: &'a ClusterControllerApp,
    endpoint: &str,
) -> Option<&'a NodeSnapshot> {
    app.nodes.iter().find(|node| {
        rpc_endpoint_candidates_for_node(node)
            .iter()
            .any(|candidate| candidate == endpoint)
    })
}

fn unique_remote_device_match_by_memory<'a>(
    devices: &'a [DeviceInfo],
    memory_total: u64,
) -> Option<&'a DeviceInfo> {
    let mut matches = devices.iter().filter(|candidate| candidate.memory_total == memory_total);
    let first = matches.next()?;
    matches.next().is_none().then_some(first)
}

fn rpc_device_ordinal_for_endpoint(
    node: &NodeSnapshot,
    device: &DeviceInfo,
    endpoint: &str,
) -> usize {
    node.devices
        .iter()
        .filter(|candidate| is_rpc_device(candidate))
        .filter(|candidate| rpc_endpoint_from_device(candidate).as_deref() == Some(endpoint))
        .filter(|candidate| candidate.bridge_device_index < device.bridge_device_index)
        .count()
}

fn group_device_indices(group: &ExecutionGroupInfo) -> Vec<i32> {
    group
        .devices_csv
        .split(',')
        .filter_map(|part| part.trim().parse::<i32>().ok())
        .collect()
}

fn group_visible_device_names(
    app: &ClusterControllerApp,
    node: &NodeSnapshot,
    group: &ExecutionGroupInfo,
    telemetry: Option<&TelemetrySnapshot>,
) -> Vec<String> {
    let devices = raw_visible_devices_for_node(app, node, telemetry);
    group_device_indices(group)
        .into_iter()
        .filter_map(|index| {
            devices
                .iter()
                .find(|device| device.bridge_device_index == index)
                .map(|device| {
                    (
                        physical_device_key(node, device),
                        device_display_name_for_ui(app, node, device),
                    )
                })
        })
        .collect::<BTreeMap<_, _>>()
        .into_values()
        .into_iter()
        .collect()
}

fn execution_group_uses_hidden_device(
    app: &ClusterControllerApp,
    node: &NodeSnapshot,
    group: &ExecutionGroupInfo,
    telemetry: Option<&TelemetrySnapshot>,
) -> bool {
    let all_devices = telemetry
        .map(|snapshot| snapshot.devices.clone())
        .unwrap_or_else(|| node.devices.clone());
    let visible_indices = raw_visible_devices_for_node(app, node, telemetry)
        .into_iter()
        .map(|device| device.bridge_device_index)
        .collect::<BTreeSet<_>>();

    group_device_indices(group).into_iter().any(|index| {
        all_devices
            .iter()
            .find(|device| device.bridge_device_index == index)
            .is_some_and(|device| !visible_indices.contains(&device.bridge_device_index))
    })
}

fn execution_group_visible_in_ui(
    app: &ClusterControllerApp,
    node: &NodeSnapshot,
    group: &ExecutionGroupInfo,
    telemetry: Option<&TelemetrySnapshot>,
) -> bool {
    if group.id == "cluster:auto" {
        return true;
    }
    if execution_group_uses_hidden_device(app, node, group, telemetry) {
        return false;
    }
    !group_visible_device_names(app, node, group, telemetry).is_empty()
}

fn filtered_execution_groups_for_node(
    app: &ClusterControllerApp,
    node: &NodeSnapshot,
    telemetry: Option<&TelemetrySnapshot>,
) -> Vec<ExecutionGroupInfo> {
    node.execution_groups
        .iter()
        .filter(|group| execution_group_visible_in_ui(app, node, group, telemetry))
        .cloned()
        .collect()
}

fn placement_targets(app: &ClusterControllerApp) -> Vec<PlacementTargetView> {
    if !app.placement_candidates.is_empty() {
        return placement_targets_from_candidates(app);
    }

    let mut targets = Vec::new();
    let mut seen = BTreeSet::new();
    for node in &app.nodes {
        for group in filtered_execution_groups_for_node(app, node, None) {
            if group.id == "cluster:auto" {
                continue;
            }
            let device_names = group_visible_device_names(app, node, &group, None);
            if device_names.is_empty() {
                continue;
            }
            let title = if group.uses_local_split {
                format!("{}: {}", node.node.display_name, device_names.join(" + "))
            } else {
                format!("{}: {}", node.node.display_name, device_names[0].clone())
            };
            let subtitle = format!(
                "{} free of {} | {}",
                format_mib_from_bytes(group.memory_free),
                format_mib_from_bytes(group.memory_total),
                group.backend_summary
            );
            let dedupe_key = format!("{}|{}", node.control_addr, title.to_ascii_lowercase());
            if !seen.insert(dedupe_key) {
                continue;
            }
            targets.push(PlacementTargetView {
                owner_control_addr: node.control_addr.clone(),
                owner_display_name: node.node.display_name.clone(),
                execution_group_id: group.id.clone(),
                rpc_servers: String::new(),
                title,
                subtitle,
                estimated_required_bytes: 0,
                estimated_group_free_bytes: group.memory_free,
                ready_now: true,
                requires_eviction: false,
            });
        }
    }
    targets.sort_by(|lhs, rhs| {
        lhs.owner_display_name
            .cmp(&rhs.owner_display_name)
            .then(lhs.title.cmp(&rhs.title))
    });
    targets
}

fn placement_targets_from_candidates(app: &ClusterControllerApp) -> Vec<PlacementTargetView> {
    let mut targets = Vec::new();
    let mut seen = BTreeSet::new();
    for plan in &app.placement_candidates {
        let title = placement_target_title_for_plan(app, plan);
        let subtitle = placement_target_subtitle_for_plan(plan);
        let dedupe_key = format!(
            "{}|{}|{}",
            plan.owner_control_addr, plan.execution_group_id, plan.rpc_servers
        );
        if !seen.insert(dedupe_key) {
            continue;
        }
        targets.push(PlacementTargetView {
            owner_control_addr: plan.owner_control_addr.clone(),
            owner_display_name: plan.owner_display_name.clone(),
            execution_group_id: plan.execution_group_id.clone(),
            rpc_servers: plan.rpc_servers.clone(),
            title,
            subtitle,
            estimated_required_bytes: plan.estimated_required_bytes,
            estimated_group_free_bytes: plan.estimated_group_free_bytes,
            ready_now: plan.ready_now,
            requires_eviction: plan.requires_eviction,
        });
    }
    targets
}

fn highlighted_placement_targets(targets: &[PlacementTargetView]) -> Vec<PlacementTargetView> {
    if targets.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    out.push(targets[0].clone());

    let mut saw_local_split =
        targets[0].execution_group_id.contains("split") && targets[0].rpc_servers.trim().is_empty();
    let mut saw_multi_node = !targets[0].rpc_servers.trim().is_empty();
    let mut saw_single = !targets[0].execution_group_id.contains("split")
        && targets[0].rpc_servers.trim().is_empty();

    for target in targets.iter().skip(1) {
        let is_multi = !target.rpc_servers.trim().is_empty();
        let is_local_split = target.execution_group_id.contains("split") && !is_multi;
        let is_single = !target.execution_group_id.contains("split") && !is_multi;
        let should_take = (is_multi && !saw_multi_node)
            || (is_local_split && !saw_local_split)
            || (is_single && !saw_single);
        if should_take {
            out.push(target.clone());
            saw_multi_node |= is_multi;
            saw_local_split |= is_local_split;
            saw_single |= is_single;
        }
        if out.len() >= 4 {
            break;
        }
    }

    for target in targets.iter().skip(1) {
        if out.len() >= 4 {
            break;
        }
        let exists = out.iter().any(|existing| {
            existing.owner_control_addr == target.owner_control_addr
                && existing.execution_group_id == target.execution_group_id
                && existing.rpc_servers == target.rpc_servers
        });
        if !exists {
            out.push(target.clone());
        }
    }

    out
}

fn placement_target_can_load(target: &PlacementTargetView) -> bool {
    let _ = target;
    true
}

fn manual_owner_choice_key(choice: &ManualOwnerChoice) -> String {
    format!(
        "{}|{}",
        choice.owner_control_addr, choice.bridge_device_index
    )
}

fn manual_device_source_key(
    source_node_id: &str,
    source_control_addr: &str,
    source_bridge_device_index: i32,
    rpc_device: bool,
) -> String {
    let source = if !source_node_id.trim().is_empty() {
        format!("node:{}", source_node_id.trim())
    } else {
        format!("addr:{}", source_control_addr.trim())
    };
    format!(
        "{source}|device:{source_bridge_device_index}|rpc:{}",
        if rpc_device { 1 } else { 0 }
    )
}

fn manual_device_choice_key(choice: &ManualDeviceChoice) -> String {
    manual_device_source_key(
        &choice.source_node_id,
        &choice.source_control_addr,
        choice.source_bridge_device_index,
        choice.rpc_device,
    )
}

fn manual_device_choice_display_label(choice: &ManualDeviceChoice) -> String {
    if choice.source_display_name.trim().is_empty() {
        choice.device_label.clone()
    } else {
        format!("{} | {}", choice.source_display_name, choice.device_label)
    }
}

fn manual_allocation_choice_key(
    row: &crate::cluster_api::ManualDeviceAllocation,
) -> Option<String> {
    (row.source_bridge_device_index >= 0
        && (!row.source_node_id.trim().is_empty() || !row.source_control_addr.trim().is_empty()))
    .then(|| {
        manual_device_source_key(
            &row.source_node_id,
            &row.source_control_addr,
            row.source_bridge_device_index,
            row.rpc_device,
        )
    })
}

fn manual_allocation_matches_choice(
    row: &crate::cluster_api::ManualDeviceAllocation,
    choice: &ManualDeviceChoice,
) -> bool {
    if let Some(key) = manual_allocation_choice_key(row) {
        return key == manual_device_choice_key(choice);
    }
    let row_label = normalized_manual_device_label(&row.device_label);
    row.bridge_device_index == choice.bridge_device_index
        && (row_label.is_empty()
            || normalized_manual_device_label(&choice.device_label) == row_label)
        && row.rpc_device == choice.rpc_device
}

fn artifact_is_available_on_owner(
    availability: &ClusterModelArtifactInfo,
    owner_control_addr: &str,
) -> bool {
    availability
        .available_on
        .iter()
        .any(|location| location.control_addr == owner_control_addr)
}

fn selected_artifact_missing_count_for_owner(
    app: &ClusterControllerApp,
    owner_control_addr: &str,
) -> Option<usize> {
    let details = app.selected_model_package_detail()?;
    let mut required_files = 0usize;
    let mut missing_files = 0usize;

    if let Some(selected_model_path) = app.selected_model_file_path.as_deref() {
        required_files += 1;
        let model_available = details
            .model_file_availability
            .iter()
            .find(|entry| entry.artifact.relative_path == selected_model_path)
            .is_some_and(|entry| artifact_is_available_on_owner(entry, owner_control_addr));
        if !model_available {
            missing_files += 1;
        }
    }

    if app.instance_model_kind == "vision" {
        if let Some(selected_mmproj_path) = app.selected_mmproj_file_path.as_deref() {
            required_files += 1;
            let mmproj_available = details
                .mmproj_file_availability
                .iter()
                .find(|entry| entry.artifact.relative_path == selected_mmproj_path)
                .is_some_and(|entry| artifact_is_available_on_owner(entry, owner_control_addr));
            if !mmproj_available {
                missing_files += 1;
            }
        }
    }

    if required_files == 0 {
        None
    } else {
        Some(missing_files)
    }
}

fn owner_artifact_availability_badge(
    app: &ClusterControllerApp,
    owner_control_addr: &str,
) -> Option<OwnerArtifactAvailabilityBadge> {
    let missing_files = selected_artifact_missing_count_for_owner(app, owner_control_addr)?;
    if missing_files == 0 {
        Some(OwnerArtifactAvailabilityBadge {
            label: "available".to_string(),
            fill: egui::Color32::from_rgb(220, 252, 231),
            text: egui::Color32::from_rgb(21, 128, 61),
        })
    } else {
        Some(OwnerArtifactAvailabilityBadge {
            label: if missing_files == 1 {
                "1 file needs transfer".to_string()
            } else {
                format!("{missing_files} files need transfer")
            },
            fill: egui::Color32::from_rgb(254, 249, 195),
            text: egui::Color32::from_rgb(161, 98, 7),
        })
    }
}

fn owner_transfer_action_label(
    app: &ClusterControllerApp,
    owner_control_addr: &str,
) -> Option<String> {
    let package = app.selected_model_package()?;
    let details = app.selected_model_package_detail()?;
    let relative_paths = app.selected_primary_transfer_relative_paths(package);
    if relative_paths.is_empty() {
        return None;
    }
    let local_control_addr = app.host.control_addr();
    let owner_display_name = app.node_display_name_for_control_addr(owner_control_addr);
    if owner_control_addr == local_control_addr {
        return Some("Retrieve to this machine".to_string());
    }

    let all_missing_files_available_locally = relative_paths
        .iter()
        .filter_map(|relative_path| {
            details
                .model_file_availability
                .iter()
                .find(|entry| entry.artifact.relative_path == *relative_path)
                .or_else(|| {
                    details
                        .mmproj_file_availability
                        .iter()
                        .find(|entry| entry.artifact.relative_path == *relative_path)
                })
        })
        .filter(|availability| {
            !availability
                .available_on
                .iter()
                .any(|node| node.control_addr == owner_control_addr)
        })
        .all(|availability| {
            availability
                .available_on
                .iter()
                .any(|node| node.control_addr == local_control_addr)
        });
    if all_missing_files_available_locally {
        Some(format!("Upload to {}", owner_display_name))
    } else {
        Some(format!("Copy selected files to {}", owner_display_name))
    }
}

fn render_combo_choice_with_badge(
    ui: &mut egui::Ui,
    selected: bool,
    label: impl Into<egui::WidgetText>,
    badge: Option<&OwnerArtifactAvailabilityBadge>,
) -> bool {
    let mut clicked = false;
    ui.horizontal_wrapped(|ui| {
        if ui.selectable_label(selected, label).clicked() {
            clicked = true;
        }
        if let Some(badge) = badge {
            state_badge(ui, &badge.label, badge.fill, badge.text);
        }
    });
    clicked
}

fn apply_manual_allocation_row_from_choice(
    row: &mut crate::cluster_api::ManualDeviceAllocation,
    choice: &ManualDeviceChoice,
) {
    row.bridge_device_index = choice.bridge_device_index;
    row.device_label = choice.device_label.clone();
    row.backend = choice.backend.clone();
    row.rpc_device = choice.rpc_device;
    row.source_node_id = choice.source_node_id.clone();
    row.source_control_addr = choice.source_control_addr.clone();
    row.source_bridge_device_index = choice.source_bridge_device_index;
}

fn manual_owner_choices(app: &ClusterControllerApp) -> Vec<ManualOwnerChoice> {
    let mut out = Vec::new();
    for node in &app.nodes {
        let telemetry = app.telemetry_for_control_addr(&node.control_addr);
        for device in filtered_devices_for_node(app, node, telemetry) {
            if is_cpu_device(&device) || is_rpc_device(&device) {
                continue;
            }
            out.push(ManualOwnerChoice {
                owner_control_addr: node.control_addr.clone(),
                owner_display_name: node.node.display_name.clone(),
                bridge_device_index: device.bridge_device_index,
                device_label: device_display_name_for_ui(app, node, &device),
                backend: device.backend.clone(),
                memory_free: device.memory_free,
                memory_total: device.memory_total,
            });
        }
    }
    out.sort_by(|lhs, rhs| {
        lhs.owner_display_name
            .cmp(&rhs.owner_display_name)
            .then(rhs.memory_total.cmp(&lhs.memory_total))
            .then(lhs.device_label.cmp(&rhs.device_label))
            .then(lhs.bridge_device_index.cmp(&rhs.bridge_device_index))
    });
    out
}

fn manual_selected_remote_peers(
    app: &ClusterControllerApp,
    owner_control_addr: &str,
) -> BTreeSet<String> {
    let selected = app.normalize_visible_node_addr_selection(&app.selected_rpc_peer_addrs);
    if selected.is_empty() {
        return app.all_visible_rpc_peer_control_addrs_for_owner(owner_control_addr);
    }
    let owner_canonical = lookup_node_for_addr(app, owner_control_addr)
        .map(|node| node.control_addr.clone())
        .unwrap_or_else(|| owner_control_addr.to_string());
    app.nodes
        .iter()
        .filter(|node| node.control_addr != owner_canonical)
        .filter(|node| node.rpc_running)
        .filter(|node| ClusterControllerApp::addr_selection_contains_node(&selected, node))
        .map(|node| node.control_addr.clone())
        .collect()
}

fn manual_choice_from_source_device(
    node: &NodeSnapshot,
    device: &DeviceInfo,
    rpc_device: bool,
) -> ManualDeviceChoice {
    ManualDeviceChoice {
        source_node_id: node.node.node_id.clone(),
        source_control_addr: node.control_addr.clone(),
        source_display_name: device_source_display_name(node),
        source_bridge_device_index: device.bridge_device_index,
        bridge_device_index: device.bridge_device_index,
        device_label: device_display_name(node, device),
        backend: device.backend.clone(),
        memory_free: device.memory_free,
        memory_total: device.memory_total,
        rpc_device,
    }
}

fn manual_preview_owner_snapshot<'a>(app: &'a ClusterControllerApp) -> Option<&'a NodeSnapshot> {
    let owner_control_addr = app.manual_owner_control_addr()?;
    app.preview_node
        .as_ref()
        .filter(|node| lookup_node_for_addr(app, owner_control_addr).is_some_and(|current| current.control_addr == node.control_addr))
        .or_else(|| lookup_node_for_addr(app, owner_control_addr))
}

fn manual_device_choices(app: &ClusterControllerApp) -> Vec<ManualDeviceChoice> {
    let Some(owner_control_addr) = app.manual_owner_control_addr() else {
        return Vec::new();
    };
    let Some(owner_snapshot) = lookup_node_for_addr(app, owner_control_addr) else {
        return Vec::new();
    };
    let owner_telemetry = app.telemetry_for_control_addr(&owner_snapshot.control_addr);
    let mut out = filtered_devices_for_node(app, owner_snapshot, owner_telemetry)
        .into_iter()
        .filter(|device| !is_cpu_device(device) && !is_rpc_device(device))
        .map(|device| manual_choice_from_source_device(owner_snapshot, &device, false))
        .collect::<Vec<_>>();

    for remote_control_addr in manual_selected_remote_peers(app, owner_control_addr) {
        let Some(remote_node) = lookup_node_for_addr(app, &remote_control_addr) else {
            continue;
        };
        let telemetry = app.telemetry_for_control_addr(&remote_node.control_addr);
        for device in filtered_devices_for_node(app, remote_node, telemetry)
            .into_iter()
            .filter(|device| !is_cpu_device(device) && !is_rpc_device(device))
        {
            out.push(manual_choice_from_source_device(remote_node, &device, true));
        }
    }
    out.sort_by(|lhs, rhs| {
        lhs.rpc_device
            .cmp(&rhs.rpc_device)
            .then(lhs.source_display_name.cmp(&rhs.source_display_name))
            .then(rhs.memory_total.cmp(&lhs.memory_total))
            .then(lhs.device_label.cmp(&rhs.device_label))
            .then(lhs.source_bridge_device_index.cmp(&rhs.source_bridge_device_index))
    });
    out
}

fn manual_runtime_device_choices(app: &ClusterControllerApp) -> Vec<ManualDeviceChoice> {
    let Some(owner_snapshot) = manual_preview_owner_snapshot(app) else {
        return Vec::new();
    };
    let devices = dedupe_visible_devices(
        owner_snapshot,
        raw_visible_devices_for_node(app, owner_snapshot, None),
    )
    .into_iter()
    .filter(|device| !is_cpu_device(device))
    .collect::<Vec<_>>();
    let mut out = devices
        .into_iter()
        .map(|device| {
            let rpc_device = is_rpc_device(&device);
            let (
                source_node_id,
                source_control_addr,
                source_display_name,
                source_bridge_device_index,
                device_label,
            ) = if rpc_device {
                let endpoint = rpc_endpoint_from_device(&device);
                let remote_node = endpoint
                    .as_deref()
                    .and_then(|value| lookup_node_for_rpc_endpoint(app, value));
                let matched_remote = endpoint.as_deref().and_then(|value| {
                    let remote_node = lookup_node_for_rpc_endpoint(app, value)?;
                    let remote_devices = filtered_devices_for_node(
                        app,
                        remote_node,
                        app.telemetry_for_control_addr(&remote_node.control_addr),
                    )
                    .into_iter()
                    .filter(|candidate| !is_cpu_device(candidate) && !is_rpc_device(candidate))
                    .collect::<Vec<_>>();
                    if let Some(exact) =
                        unique_remote_device_match_by_memory(&remote_devices, device.memory_total)
                    {
                        return Some(exact.clone());
                    }
                    let rpc_ordinal =
                        rpc_device_ordinal_for_endpoint(owner_snapshot, &device, value);
                    let mut sorted_remote_devices = remote_devices;
                    sorted_remote_devices.sort_by(|lhs, rhs| {
                        lhs.bridge_device_index
                            .cmp(&rhs.bridge_device_index)
                            .then(lhs.memory_total.cmp(&rhs.memory_total))
                            .then(lhs.name.cmp(&rhs.name))
                    });
                    sorted_remote_devices.get(rpc_ordinal).cloned()
                });
                (
                    remote_node
                        .map(|node| node.node.node_id.clone())
                        .unwrap_or_default(),
                    remote_node
                        .map(|node| node.control_addr.clone())
                        .unwrap_or_default(),
                    remote_node
                        .map(device_source_display_name)
                        .unwrap_or_else(|| device_display_name_for_ui(app, owner_snapshot, &device)),
                    matched_remote
                        .as_ref()
                        .map(|remote| remote.bridge_device_index)
                        .unwrap_or(-1),
                    matched_remote
                        .as_ref()
                        .zip(remote_node)
                        .map(|(remote, node)| device_display_name(node, remote))
                        .unwrap_or_else(|| device_display_name_for_ui(app, owner_snapshot, &device)),
                )
            } else {
                (
                    owner_snapshot.node.node_id.clone(),
                    owner_snapshot.control_addr.clone(),
                    device_source_display_name(owner_snapshot),
                    device.bridge_device_index,
                    device_display_name(owner_snapshot, &device),
                )
            };
            ManualDeviceChoice {
                source_node_id,
                source_control_addr,
                source_display_name,
                source_bridge_device_index,
                bridge_device_index: device.bridge_device_index,
                device_label,
                backend: device.backend.clone(),
                memory_free: device.memory_free,
                memory_total: device.memory_total,
                rpc_device,
            }
        })
        .collect::<Vec<_>>();
    out.sort_by(|lhs, rhs| {
        lhs.rpc_device
            .cmp(&rhs.rpc_device)
            .then(rhs.memory_total.cmp(&lhs.memory_total))
            .then(lhs.source_display_name.cmp(&rhs.source_display_name))
            .then(lhs.device_label.cmp(&rhs.device_label))
            .then(lhs.bridge_device_index.cmp(&rhs.bridge_device_index))
    });
    out
}

fn normalized_manual_device_label(value: &str) -> String {
    value.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

pub(crate) fn resolve_live_manual_device_allocations(
    app: &ClusterControllerApp,
) -> (Vec<crate::cluster_api::ManualDeviceAllocation>, BTreeSet<String>) {
    let choices = manual_runtime_device_choices(app);
    if choices.is_empty() {
        return (app.create_params.manual_device_allocations.clone(), BTreeSet::new());
    }

    let owner_control_addr = app.manual_owner_control_addr().map(str::to_string);
    let mut used_choice_keys = BTreeSet::new();
    let mut selected_remote_control_addrs = BTreeSet::new();
    let mut resolved = Vec::with_capacity(app.create_params.manual_device_allocations.len());

    for (row_index, row) in app.create_params.manual_device_allocations.iter().enumerate() {
        let row_label = normalized_manual_device_label(&row.device_label);

        let mut matched = choices
            .iter()
            .find(|choice| manual_allocation_matches_choice(row, choice))
            .cloned();

        if matched.is_none() && !row_label.is_empty() {
            let label_matches = choices
                .iter()
                .filter(|choice| normalized_manual_device_label(&choice.device_label) == row_label)
                .cloned()
                .collect::<Vec<_>>();
            let rpc_filtered = label_matches
                .iter()
                .filter(|choice| choice.rpc_device == row.rpc_device)
                .cloned()
                .collect::<Vec<_>>();
            let owner_filtered = rpc_filtered
                .iter()
                .filter(|choice| {
                    row_index != 0
                        || owner_control_addr.as_deref()
                            == Some(choice.source_control_addr.as_str())
                })
                .cloned()
                .collect::<Vec<_>>();
            let candidates = if !owner_filtered.is_empty() {
                owner_filtered
            } else if !rpc_filtered.is_empty() {
                rpc_filtered
            } else {
                label_matches
            };
            let unused = candidates
                .iter()
                .filter(|choice| !used_choice_keys.contains(&manual_device_choice_key(choice)))
                .cloned()
                .collect::<Vec<_>>();
            matched = unused
                .into_iter()
                .next()
                .or_else(|| candidates.into_iter().next());
        }

        if matched.is_none() && row_index == 0 {
            matched = choices
                .iter()
                .find(|choice| {
                    !choice.rpc_device
                        && owner_control_addr.as_deref()
                            == Some(choice.source_control_addr.as_str())
                })
                .cloned();
        }

        if let Some(choice) = matched {
            used_choice_keys.insert(manual_device_choice_key(&choice));
            if choice.rpc_device {
                selected_remote_control_addrs.insert(choice.source_control_addr.clone());
            }
            resolved.push(crate::cluster_api::ManualDeviceAllocation {
                bridge_device_index: choice.bridge_device_index,
                device_label: choice.device_label,
                backend: choice.backend,
                layer_count: row.layer_count,
                rpc_device: choice.rpc_device,
                source_node_id: choice.source_node_id,
                source_control_addr: choice.source_control_addr,
                source_bridge_device_index: choice.source_bridge_device_index,
            });
        } else {
            resolved.push(row.clone());
        }
    }

    (resolved, selected_remote_control_addrs)
}

fn set_manual_primary_owner(app: &mut ClusterControllerApp, choice: &ManualOwnerChoice) {
    let preserved_layers = app
        .create_params
        .manual_device_allocations
        .first()
        .map(|row| row.layer_count)
        .unwrap_or(0);
    app.create_params.preferred_owner_control_addr = Some(choice.owner_control_addr.clone());
    app.create_params.execution_group_id = "cluster:manual".to_string();
    app.create_params.rpc_servers = None;
    app.create_params.manual_devices_csv = None;
    app.create_params.manual_tensor_split = None;
    app.create_params.manual_device_allocations =
        vec![crate::cluster_api::ManualDeviceAllocation {
            bridge_device_index: choice.bridge_device_index,
            device_label: choice.device_label.clone(),
            backend: choice.backend.clone(),
            layer_count: preserved_layers,
            rpc_device: false,
            source_node_id: lookup_node_for_addr(app, &choice.owner_control_addr)
                .map(|node| node.node.node_id.clone())
                .unwrap_or_default(),
            source_control_addr: choice.owner_control_addr.clone(),
            source_bridge_device_index: choice.bridge_device_index,
        }];
    app.sync_selected_model_package();
    app.selected_control_addr = Some(choice.owner_control_addr.clone());
    app.selected_rpc_peer_addrs.clear();
    let _ = app.refresh_selected_preview();
    app.last_plan = None;
}

fn ensure_manual_allocation_seed(
    app: &mut ClusterControllerApp,
    targets: &[PlacementTargetView],
    selected_target: Option<&PlacementTargetView>,
) {
    if !app.create_params.manual_device_allocations.is_empty()
        && app.manual_owner_control_addr().is_some()
    {
        let owner_control_addr = app.manual_owner_control_addr().unwrap_or_default().to_string();
        let visible_remote_peers = app.all_visible_rpc_peer_control_addrs_for_owner(&owner_control_addr);
        if !visible_remote_peers.is_empty() && app.selected_rpc_peer_addrs.is_empty() {
            app.selected_rpc_peer_addrs = visible_remote_peers;
        }
        return;
    }
    let owner_choices = manual_owner_choices(app);
    if owner_choices.is_empty() {
        return;
    }

    let seeded = selected_target
        .and_then(|target| {
            owner_choices
                .iter()
                .find(|choice| choice.owner_control_addr == target.owner_control_addr)
        })
        .or_else(|| {
            targets.first().and_then(|target| {
                owner_choices
                    .iter()
                    .find(|choice| choice.owner_control_addr == target.owner_control_addr)
            })
        })
        .unwrap_or(&owner_choices[0])
        .clone();
    set_manual_primary_owner(app, &seeded);
    app.selected_rpc_peer_addrs = app
        .all_visible_rpc_peer_control_addrs_for_owner(&seeded.owner_control_addr);
    let _ = app.refresh_selected_preview();
}

fn manual_allocation_summary(
    app: &ClusterControllerApp,
    runtime_estimate: Option<&RuntimeVramEstimate>,
    total_layers: Option<u32>,
) -> ManualAllocationSummary {
    let single_device_full_offload = manual_single_device_full_offload(app);
    let choices = manual_device_choices(app);
    let live_by_key = choices
        .iter()
        .map(|choice| (manual_device_choice_key(choice), choice.clone()))
        .collect::<BTreeMap<_, _>>();
    let allocated_layers = app
        .create_params
        .manual_device_allocations
        .iter()
        .map(|row| row.layer_count)
        .sum::<u32>();
    let total_assigned = if single_device_full_offload {
        total_layers.unwrap_or(1)
    } else {
        app.create_params
            .manual_device_allocations
            .iter()
            .map(|row| {
                total_layers
                    .map(|limit| row.layer_count.min(limit))
                    .unwrap_or(row.layer_count)
            })
            .sum::<u32>()
    };

    let shape_label = if app.create_params.manual_device_allocations.len() <= 1 {
        "single GPU"
    } else if app
        .create_params
        .manual_device_allocations
        .iter()
        .skip(1)
        .any(|row| row.rpc_device)
    {
        "multi-node split"
    } else {
        "same-host split"
    };

    let Some(estimate) = runtime_estimate else {
        return ManualAllocationSummary {
            allocated_layers,
            total_layers,
            shape_label,
            fit: None,
            row_estimates: Vec::new(),
        };
    };

    let non_mmproj_bytes = estimate
        .required_gpu_bytes
        .saturating_sub(estimate.mmproj_gpu_bytes);
    let mut row_estimates = Vec::new();
    let mut fit = Some(ManualAllocationFit::ReadyNow);
    for (index, row) in app
        .create_params
        .manual_device_allocations
        .iter()
        .enumerate()
    {
        let live = manual_allocation_choice_key(row).and_then(|key| live_by_key.get(&key));
        let effective_layers = if single_device_full_offload && index == 0 {
            total_layers.unwrap_or(1)
        } else {
            total_layers
                .map(|limit| row.layer_count.min(limit))
                .unwrap_or(row.layer_count)
        };
        let mut estimated_bytes = if total_assigned == 0 {
            0
        } else {
            ((non_mmproj_bytes as f64) * (effective_layers as f64) / (total_assigned as f64))
                .round() as u64
        };
        if index == 0 {
            estimated_bytes = estimated_bytes.saturating_add(estimate.mmproj_gpu_bytes);
        }
        let memory_free = live.map(|choice| choice.memory_free).unwrap_or(0);
        let memory_total = live.map(|choice| choice.memory_total).unwrap_or(0);
        let available = live.is_some();
        if !available || memory_total == 0 || estimated_bytes > memory_total {
            fit = Some(ManualAllocationFit::Insufficient);
        } else if fit == Some(ManualAllocationFit::ReadyNow) && estimated_bytes > memory_free {
            fit = Some(ManualAllocationFit::EvictionNeeded);
        }
        row_estimates.push(ManualAllocationEstimate {
            device_label: live
                .map(|choice| choice.device_label.clone())
                .unwrap_or_else(|| row.device_label.clone()),
            backend: live
                .map(|choice| choice.backend.clone())
                .unwrap_or_else(|| row.backend.clone()),
            layer_count: row.layer_count,
            estimated_bytes,
            memory_free,
            memory_total,
            rpc_device: live
                .map(|choice| choice.rpc_device)
                .unwrap_or(row.rpc_device),
            available,
        });
    }

    ManualAllocationSummary {
        allocated_layers,
        total_layers,
        shape_label,
        fit,
        row_estimates,
    }
}

fn manual_layer_slider_max(
    total_layers: Option<u32>,
    allocations: &[crate::cluster_api::ManualDeviceAllocation],
) -> i32 {
    if let Some(total_layers) = total_layers {
        return i32::try_from(total_layers).unwrap_or(i32::MAX);
    }
    let current_max = allocations
        .iter()
        .map(|row| row.layer_count)
        .max()
        .unwrap_or(0);
    let dynamic_limit = current_max.saturating_add(64).max(128);
    i32::try_from(dynamic_limit).unwrap_or(i32::MAX)
}

fn manual_single_device_full_offload(app: &ClusterControllerApp) -> bool {
    app.create_params.manual_device_allocations.len() == 1
        && app
            .create_params
            .manual_device_allocations
            .first()
            .map(|row| row.layer_count == 0)
            .unwrap_or(false)
}

fn placement_target_choice_key(target: &PlacementTargetView) -> String {
    format!(
        "{}|{}|{}",
        target.owner_control_addr, target.execution_group_id, target.rpc_servers
    )
}

fn render_target_distribution_preview(
    app: &ClusterControllerApp,
    ui: &mut egui::Ui,
    target: &PlacementTargetView,
    estimate: &RuntimeVramEstimate,
) {
    let Some(owner) = lookup_node_for_addr(app, &target.owner_control_addr) else {
        ui.label("Selected owner node is not visible in the current cluster state.");
        return;
    };
    let telemetry = app.telemetry_for_control_addr(&owner.control_addr);
    let group = filtered_execution_groups_for_node(app, owner, telemetry)
        .into_iter()
        .find(|group| group.id == target.execution_group_id);
    let Some(group) = group else {
        ui.label("The selected execution group is not visible on the owner node.");
        return;
    };

    let mut devices = raw_visible_devices_for_node(app, owner, telemetry)
        .into_iter()
        .filter(|device| group_contains_device_index(&group, device.bridge_device_index))
        .filter(|device| !is_cpu_device(device))
        .collect::<Vec<_>>();
    if devices.is_empty() {
        ui.label("No GPU devices are exposed for the selected target.");
        return;
    }
    devices.sort_by_key(|device| device.bridge_device_index);

    let total_memory = devices
        .iter()
        .map(|device| device.memory_total.max(1))
        .sum::<u64>()
        .max(1);
    let total_layers = estimate.detected_layer_count.unwrap_or(0);
    let layers_on_gpu = estimate.layers_on_gpu.unwrap_or(total_layers);
    let non_mmproj_bytes = estimate
        .required_gpu_bytes
        .saturating_sub(estimate.mmproj_gpu_bytes);

    let mut remaining_layers = layers_on_gpu;
    for (index, device) in devices.iter().enumerate() {
        let share = device.memory_total.max(1) as f64 / total_memory as f64;
        let layer_share = if index + 1 == devices.len() {
            remaining_layers
        } else {
            ((layers_on_gpu as f64) * share).round() as u32
        }
        .min(remaining_layers);
        remaining_layers = remaining_layers.saturating_sub(layer_share);

        let mut device_bytes = ((non_mmproj_bytes as f64) * share).round() as u64;
        if index == 0 {
            device_bytes = device_bytes.saturating_add(estimate.mmproj_gpu_bytes);
        }
        let used_ratio = memory_ratio(device_bytes, device.memory_total);

        ui.label(
            egui::RichText::new(format!(
                "{} [{}]",
                device_display_name_for_ui(app, owner, device),
                device.backend
            ))
            .strong(),
        );
        ui.add(
            egui::ProgressBar::new(used_ratio)
                .desired_width(ui.available_width())
                .text(format!(
                    "{} est. of {}",
                    format_mib(device_bytes),
                    format_mib(device.memory_total)
                )),
        );
        if total_layers > 0 {
            ui.label(format!(
                "{} / {} GPU layers | {} free right now",
                layer_share,
                layers_on_gpu,
                format_mib(device.memory_free)
            ));
        } else {
            ui.label(format!("{} free right now", format_mib(device.memory_free)));
        }
        if index == 0 && estimate.mmproj_gpu_bytes > 0 {
            muted_label(ui, "Includes the full MMProj on the primary GPU.");
        }
        ui.add_space(6.0);
    }
}

fn selected_target_label(
    app: &ClusterControllerApp,
    targets: &[PlacementTargetView],
    auto_target_label: &str,
) -> String {
    if app.create_params.execution_group_id.trim().is_empty()
        || app.create_params.execution_group_id == "cluster:auto"
    {
        return auto_target_label.to_string();
    }
    let preferred_owner = app.create_params.preferred_owner_control_addr.as_deref();
    targets
        .iter()
        .find(|target| {
            Some(target.owner_control_addr.as_str()) == preferred_owner
                && target.execution_group_id == app.create_params.execution_group_id
                && target.rpc_servers == app.create_params.rpc_servers.clone().unwrap_or_default()
        })
        .map(|target| target.title.clone())
        .unwrap_or_else(|| app.create_params.execution_group_id.clone())
}

fn lookup_node_for_addr<'a>(app: &'a ClusterControllerApp, addr: &str) -> Option<&'a NodeSnapshot> {
    app.nodes.iter().find(|node| {
        node.control_addr == addr
            || node
                .advertised_control_addr
                .as_deref()
                .is_some_and(|value| value == addr)
            || node.known_control_addrs.iter().any(|value| value == addr)
    })
}

fn display_control_paths(current: &str, known: &[String]) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for value in std::iter::once(current.to_string()).chain(known.iter().cloned()) {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            continue;
        }
        let normalized = trimmed.to_string();
        if seen.insert(normalized.clone()) {
            out.push(normalized);
        }
    }
    out
}

fn render_control_paths(ui: &mut egui::Ui, label: &str, addrs: &[String]) {
    if addrs.is_empty() {
        return;
    }
    ui.label(label);
    for addr in addrs {
        wrapped_monospace(ui, addr);
    }
}

fn best_target_for_owner_control_addr<'a>(
    targets: &'a [PlacementTargetView],
    owner_control_addr: &str,
) -> Option<&'a PlacementTargetView> {
    targets
        .iter()
        .find(|target| target.owner_control_addr == owner_control_addr)
}

fn set_auto_placement_target(app: &mut ClusterControllerApp) {
    app.create_params.preferred_owner_control_addr = None;
    app.create_params.execution_group_id = "cluster:auto".to_string();
    app.create_params.rpc_servers = None;
    app.sync_selected_model_package();
    app.set_selected_rpc_workers_from_csv(None);
}

fn apply_placement_target(app: &mut ClusterControllerApp, target: &PlacementTargetView) {
    app.create_params.preferred_owner_control_addr = Some(target.owner_control_addr.clone());
    app.create_params.execution_group_id = target.execution_group_id.clone();
    app.create_params.rpc_servers = if target.rpc_servers.trim().is_empty() {
        None
    } else {
        Some(target.rpc_servers.clone())
    };
    app.sync_selected_model_package();
    app.selected_control_addr = Some(target.owner_control_addr.clone());
    let rpc_servers = app.create_params.rpc_servers.clone();
    app.set_selected_rpc_workers_from_csv(rpc_servers.as_deref());
    let _ = app.refresh_selected_preview();
}

fn dedupe_visible_devices(node: &NodeSnapshot, devices: Vec<DeviceInfo>) -> Vec<DeviceInfo> {
    let mut best_by_key = BTreeMap::new();
    for device in devices {
        let key = physical_device_key(node, &device);
        match best_by_key.get_mut(&key) {
            Some(existing) => {
                if device_backend_rank(&device) < device_backend_rank(existing)
                    || (device_backend_rank(&device) == device_backend_rank(existing)
                        && device.memory_total > existing.memory_total)
                {
                    *existing = device;
                }
            }
            None => {
                best_by_key.insert(key, device);
            }
        }
    }
    best_by_key.into_values().collect()
}

fn placement_target_title_for_plan(app: &ClusterControllerApp, plan: &PlacementPlan) -> String {
    if !plan.display_label.trim().is_empty() {
        return plan.display_label.clone();
    }

    let owner = app.nodes.iter().find(|node| {
        node.control_addr == plan.owner_control_addr
            || node.advertised_control_addr.as_deref() == Some(plan.owner_control_addr.as_str())
    });

    let mut labels = local_labels_for_plan(app, owner, plan);
    labels.extend(remote_labels_for_plan(app, plan));
    labels.retain(|label| !label.trim().is_empty());
    labels = labels
        .into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    if labels.is_empty() {
        return format!("{}: {}", plan.owner_display_name, plan.execution_group_id);
    }
    format!("{}: {}", plan.owner_display_name, labels.join(" + "))
}

fn placement_target_subtitle_for_plan(plan: &PlacementPlan) -> String {
    let mut details = vec![placement_strategy_label(plan.strategy).to_string()];
    if plan.estimated_required_bytes > 0 {
        details.push(format!(
            "needs {}",
            format_mib_from_bytes(plan.estimated_required_bytes)
        ));
    }
    if plan.estimated_group_free_bytes > 0 {
        details.push(format!(
            "{} free",
            format_mib_from_bytes(plan.estimated_group_free_bytes)
        ));
    }
    if !plan.rpc_servers.trim().is_empty() {
        details.push(format!(
            "{} remote node{}",
            plan.remote_node_count,
            if plan.remote_node_count == 1 { "" } else { "s" }
        ));
    }
    details.push(if plan.ready_now {
        "ready now".to_string()
    } else if plan.requires_eviction {
        "load_on_demand eviction needed".to_string()
    } else {
        "not enough free memory right now".to_string()
    });
    details.join(" | ")
}

fn local_labels_for_plan(
    app: &ClusterControllerApp,
    owner: Option<&NodeSnapshot>,
    plan: &PlacementPlan,
) -> Vec<String> {
    let Some(owner) = owner else {
        return Vec::new();
    };

    let visible_devices = raw_visible_devices_for_node(
        app,
        owner,
        app.telemetry_for_control_addr(&owner.control_addr),
    );
    if plan.execution_group_id.contains("gpu-all") {
        return visible_devices
            .into_iter()
            .filter(|device| !is_rpc_device(device) && !is_cpu_device(device))
            .map(|device| {
                (
                    physical_device_key(owner, &device),
                    device_display_name_for_ui(app, owner, &device),
                )
            })
            .collect::<BTreeMap<_, _>>()
            .into_values()
            .collect();
    }

    let indices = if let Some((_, csv)) = plan.execution_group_id.split_once(':') {
        csv.split(',')
            .filter_map(|part| part.trim().parse::<i32>().ok())
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };

    indices
        .into_iter()
        .filter_map(|index| {
            visible_devices
                .iter()
                .find(|device| device.bridge_device_index == index)
                .map(|device| {
                    (
                        physical_device_key(owner, device),
                        device_display_name_for_ui(app, owner, device),
                    )
                })
        })
        .collect::<BTreeMap<_, _>>()
        .into_values()
        .collect()
}

fn remote_labels_for_plan(app: &ClusterControllerApp, plan: &PlacementPlan) -> Vec<String> {
    plan.rpc_servers
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .filter_map(|rpc_server| {
            app.nodes.iter().find(|node| {
                node.rpc_endpoint.as_deref() == Some(rpc_server)
                    || node.advertised_rpc_endpoint.as_deref() == Some(rpc_server)
            })
        })
        .map(|node| {
            let first_device = filtered_devices_for_node(
                app,
                node,
                app.telemetry_for_control_addr(&node.control_addr),
            )
            .into_iter()
            .find(|device| !is_cpu_device(device) && !is_rpc_device(device))
            .map(|device| device_display_name_for_ui(app, node, &device));
            match first_device {
                Some(device) => format!("{} {}", node.node.display_name, device),
                None => node.node.display_name.clone(),
            }
        })
        .collect()
}

fn short_device_label(value: String) -> String {
    let trimmed = value.trim();
    if trimmed.chars().count() <= 28 {
        return trimmed.to_string();
    }
    let mut output = trimmed.chars().take(28).collect::<String>();
    output.push_str("...");
    output
}

fn physical_device_key(node: &NodeSnapshot, device: &DeviceInfo) -> String {
    let display = device_display_name(node, device).to_ascii_lowercase();
    if is_cpu_device(device) {
        return format!("cpu|{display}");
    }
    if is_rpc_device(device) {
        return format!(
            "rpc|{display}|{}|{}",
            device.bridge_device_index, device.memory_total
        );
    }
    format!("{display}|{}", device.memory_total)
}

fn device_backend_rank(device: &DeviceInfo) -> i32 {
    let lowered = format!("{} {}", device.backend, device.name).to_ascii_lowercase();
    if lowered.contains("cuda") {
        0
    } else if lowered.contains("metal") {
        1
    } else if lowered.contains("vulkan") {
        2
    } else if lowered.contains("rpc") {
        3
    } else if lowered.contains("cpu") {
        5
    } else {
        4
    }
}

fn group_contains_device_index(group: &ExecutionGroupInfo, device_index: i32) -> bool {
    group_device_indices(group)
        .into_iter()
        .any(|index| index == device_index)
}

fn estimate_instance_resident_bytes(instance: &crate::cluster_api::InstanceInfo) -> u64 {
    let mut total = 0u64;
    for candidate in [&instance.model_path, &instance.mmproj_path] {
        let path = candidate.trim();
        if path.is_empty() {
            continue;
        }
        if let Ok(metadata) = fs::metadata(Path::new(path)) {
            total = total.saturating_add(metadata.len());
        }
    }
    total
}

fn render_estimated_instance_segments(
    ui: &mut egui::Ui,
    used_bytes: u64,
    instances: &[crate::cluster_api::InstanceInfo],
) {
    if used_bytes == 0 || instances.is_empty() {
        return;
    }
    let mut estimates = instances
        .iter()
        .map(|instance| {
            (
                instance.instance_id,
                estimate_instance_resident_bytes(instance),
            )
        })
        .collect::<Vec<_>>();
    let total_estimated = estimates.iter().map(|(_, bytes)| *bytes).sum::<u64>();
    if total_estimated == 0 {
        let equal = used_bytes / instances.len() as u64;
        estimates = instances
            .iter()
            .map(|instance| (instance.instance_id, equal))
            .collect::<Vec<_>>();
    }

    ui.horizontal_wrapped(|ui| {
        for instance in instances {
            let estimate = estimates
                .iter()
                .find(|(instance_id, _)| *instance_id == instance.instance_id)
                .map(|(_, value)| *value)
                .unwrap_or(0);
            let scaled = if total_estimated == 0 {
                estimate
            } else {
                ((estimate as f64 / total_estimated as f64) * used_bytes as f64) as u64
            };
            summary_pill(
                ui,
                format!(
                    "{} ~{}{}{}",
                    instance.name,
                    format_mib_from_bytes(scaled),
                    if instance.active_request_count > 0 {
                        format!(" | active {}", instance.active_request_count)
                    } else {
                        String::new()
                    },
                    if instance.queued_request_count > 0 {
                        format!(" | queued {}", instance.queued_request_count)
                    } else {
                        String::new()
                    }
                ),
                instance_chip_color(instance.instance_id),
                egui::Color32::WHITE,
            );
        }
    });
}

fn render_device_instance_summary(
    app: &ClusterControllerApp,
    ui: &mut egui::Ui,
    telemetry: &TelemetrySnapshot,
    node: &NodeSnapshot,
    device: &DeviceInfo,
) {
    let groups_for_device = filtered_execution_groups_for_node(app, node, Some(telemetry))
        .iter()
        .filter(|group| group_contains_device_index(group, device.bridge_device_index))
        .map(|group| group.id.clone())
        .collect::<Vec<_>>();
    let instances = telemetry
        .instances
        .iter()
        .filter(|instance| {
            instance.state != 0
                && groups_for_device
                    .iter()
                    .any(|group_id| group_id == &instance.execution_group_id)
        })
        .cloned()
        .collect::<Vec<_>>();
    if instances.is_empty() {
        return;
    }
    render_estimated_instance_segments(
        ui,
        device.memory_total.saturating_sub(device.memory_free),
        &instances,
    );
}

fn render_node_devices_card(
    app: &ClusterControllerApp,
    ui: &mut egui::Ui,
    node: &NodeSnapshot,
    telemetry: Option<&TelemetrySnapshot>,
) {
    card(ui, Some("Devices"), |ui| {
        let devices = filtered_devices_for_node(app, node, telemetry);
        if devices.is_empty() {
            muted_label(ui, "No visible GPU devices with the current filters.");
        }
        for device in devices {
            let used = device.memory_total.saturating_sub(device.memory_free);
            ui.label(
                egui::RichText::new(format!(
                    "{} [{}]",
                    device_display_name_for_ui(app, node, &device),
                    device.backend
                ))
                .strong(),
            );
            ui.add(
                egui::ProgressBar::new(memory_ratio(used, device.memory_total))
                    .desired_width(ui.available_width())
                    .text(format!(
                        "{} free of {}",
                        format_mib(device.memory_free),
                        format_mib(device.memory_total)
                    )),
            );
            muted_label(ui, &device.description);
            if let Some(telemetry) = telemetry {
                render_device_instance_summary(app, ui, telemetry, node, &device);
            }
            ui.add_space(6.0);
        }
    });
}

fn render_node_execution_groups_card(
    app: &ClusterControllerApp,
    ui: &mut egui::Ui,
    node: &NodeSnapshot,
    telemetry: Option<&TelemetrySnapshot>,
) {
    card(ui, Some("Execution groups"), |ui| {
        let visible_groups = filtered_execution_groups_for_node(app, node, telemetry);
        if visible_groups.is_empty() {
            muted_label(ui, "No visible placement groups with the current filters.");
        }
        for group in visible_groups {
            if group.id == "cluster:auto" {
                continue;
            }
            let used = group.memory_total.saturating_sub(group.memory_free);
            let names = group_visible_device_names(app, node, &group, telemetry);
            ui.label(
                egui::RichText::new(if names.is_empty() {
                    group.label.clone()
                } else {
                    names.join(" + ")
                })
                .strong(),
            );
            ui.label(format!("{} | {}", group.backend_summary, group.id));
            ui.add(
                egui::ProgressBar::new(memory_ratio(used, group.memory_total))
                    .desired_width(ui.available_width())
                    .text(format!(
                        "{} free of {}",
                        format_mib(group.memory_free),
                        format_mib(group.memory_total)
                    )),
            );
            let instances = telemetry
                .map(|item| item.instances.clone())
                .unwrap_or_default()
                .into_iter()
                .filter(|instance| instance.execution_group_id == group.id && instance.state != 0)
                .collect::<Vec<_>>();
            if !instances.is_empty() {
                ui.horizontal_wrapped(|ui| {
                    for instance in &instances {
                        summary_pill(
                            ui,
                            instance.name.clone(),
                            instance_chip_color(instance.instance_id),
                            egui::Color32::WHITE,
                        );
                    }
                });
                render_estimated_instance_segments(
                    ui,
                    group.memory_total.saturating_sub(group.memory_free),
                    &instances,
                );
            }
            ui.add_space(6.0);
        }
    });
}

fn instance_chip_color(instance_id: i64) -> egui::Color32 {
    const COLORS: [egui::Color32; 6] = [
        egui::Color32::from_rgb(14, 165, 233),
        egui::Color32::from_rgb(34, 197, 94),
        egui::Color32::from_rgb(249, 115, 22),
        egui::Color32::from_rgb(168, 85, 247),
        egui::Color32::from_rgb(236, 72, 153),
        egui::Color32::from_rgb(245, 158, 11),
    ];
    COLORS[(instance_id.unsigned_abs() as usize) % COLORS.len()]
}

fn summary_card(
    ui: &mut egui::Ui,
    title: &str,
    value: &str,
    subtitle: &str,
    accent: egui::Color32,
) {
    let _ = summary_card_response(ui, title, value, subtitle, accent);
}

fn summary_card_sized(
    ui: &mut egui::Ui,
    size: egui::Vec2,
    title: &str,
    value: &str,
    subtitle: &str,
    accent: egui::Color32,
) {
    let _ = summary_card_sized_response(ui, size, title, value, subtitle, accent);
}

fn summary_card_response(
    ui: &mut egui::Ui,
    title: &str,
    value: &str,
    subtitle: &str,
    accent: egui::Color32,
) -> egui::Response {
    summary_card_sized_response(
        ui,
        egui::vec2(ui.available_width(), 90.0),
        title,
        value,
        subtitle,
        accent,
    )
}

fn summary_card_sized_response(
    ui: &mut egui::Ui,
    size: egui::Vec2,
    title: &str,
    value: &str,
    subtitle: &str,
    accent: egui::Color32,
) -> egui::Response {
    let (fill, stroke, accent_text) = themed_metric_colors(ui, accent);
    let inner_margin = egui::Margin::same(8);
    let inner_size = egui::vec2(
        (size.x - inner_margin.leftf() - inner_margin.rightf()).max(0.0),
        (size.y - inner_margin.topf() - inner_margin.bottomf()).max(0.0),
    );
    ui.allocate_ui_with_layout(size, egui::Layout::top_down(egui::Align::Min), |ui| {
        egui::Frame::default()
            .fill(fill)
            .stroke(stroke)
            .corner_radius(egui::CornerRadius::same(14))
            .inner_margin(inner_margin)
            .show(ui, |ui| {
                ui.set_min_size(inner_size);
                ui.with_layout(egui::Layout::top_down(egui::Align::Min), |ui| {
                    ui.label(
                        egui::RichText::new(title)
                            .strong()
                            .size(10.5)
                            .color(ui.visuals().strong_text_color()),
                    );
                    ui.label(
                        egui::RichText::new(value)
                            .size(14.0)
                            .strong()
                            .color(accent_text),
                    );
                    muted_label(ui, subtitle);
                });
            });
    })
    .response
}

struct SummaryMetricCard {
    title: String,
    value: String,
    subtitle: String,
    accent: egui::Color32,
}

impl SummaryMetricCard {
    fn new(
        title: impl Into<String>,
        value: impl Into<String>,
        subtitle: impl Into<String>,
        accent: egui::Color32,
    ) -> Self {
        Self {
            title: title.into(),
            value: value.into(),
            subtitle: subtitle.into(),
            accent,
        }
    }
}

fn render_compact_overview_grid(ui: &mut egui::Ui, metrics: &[SummaryMetricCard]) {
    if metrics.is_empty() {
        return;
    }
    let spacing = 6.0;
    let columns = 2;
    let tile_width = ((ui.available_width() - spacing) / 2.0).max(96.0);
    let tile_size = egui::vec2(tile_width, 74.0);
    for (row_index, row) in metrics.chunks(columns).enumerate() {
        ui.horizontal(|ui| {
            for (index, metric) in row.iter().enumerate() {
                compact_overview_card(
                    ui,
                    tile_size,
                    &metric.title,
                    &metric.value,
                    &metric.subtitle,
                    metric.accent,
                );
                if index + 1 < row.len() {
                    ui.add_space(spacing);
                }
            }
            for _ in row.len()..columns {
                ui.allocate_space(tile_size);
            }
        });
        if row_index + 1 < metrics.len().div_ceil(columns) {
            ui.add_space(4.0);
        }
    }
}

fn render_summary_metric_grid(
    ui: &mut egui::Ui,
    min_card_width: f32,
    max_columns: usize,
    metrics: &[SummaryMetricCard],
) {
    if metrics.is_empty() {
        return;
    }
    let spacing = ui.spacing().item_spacing.x.max(6.0);
    let mut columns =
        ((ui.available_width() + spacing) / (min_card_width + spacing)).floor() as usize;
    columns = columns.clamp(1, max_columns.max(1)).min(metrics.len());
    let tile_width = if columns > 1 {
        ((ui.available_width() - (spacing * (columns.saturating_sub(1) as f32))) / columns as f32)
            .max(min_card_width)
    } else {
        ui.available_width().max(min_card_width)
    };
    let tile_size = egui::vec2(tile_width, 58.0);
    let total_rows = metrics.len().div_ceil(columns);
    for (row_index, row) in metrics.chunks(columns).enumerate() {
        ui.horizontal(|ui| {
            for (index, metric) in row.iter().enumerate() {
                summary_card_sized(
                    ui,
                    tile_size,
                    &metric.title,
                    &metric.value,
                    &metric.subtitle,
                    metric.accent,
                );
                if index + 1 < row.len() {
                    ui.add_space(spacing);
                }
            }
            for _ in row.len()..columns {
                ui.allocate_space(tile_size);
            }
        });
        if row_index + 1 < total_rows {
            ui.add_space(3.0);
        }
    }
}

fn compact_overview_card(
    ui: &mut egui::Ui,
    size: egui::Vec2,
    title: &str,
    value: &str,
    subtitle: &str,
    accent: egui::Color32,
) {
    let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
    let (fill, stroke, accent_text) = themed_metric_colors(ui, accent);
    let palette = controller_palette_for_ui(ui);
    ui.painter().rect(
        rect,
        egui::CornerRadius::same(14),
        fill,
        stroke,
        egui::StrokeKind::Outside,
    );

    let center_x = rect.center().x;
    let title_y = rect.top() + 10.0;
    let value_y = rect.top() + 28.0;
    let subtitle_y = rect.top() + 53.0;

    ui.painter().text(
        egui::pos2(center_x, title_y),
        egui::Align2::CENTER_TOP,
        title,
        egui::FontId::new(10.5, egui::FontFamily::Proportional),
        blend_color(ui.visuals().strong_text_color(), palette.muted_text, 0.22),
    );
    ui.painter().text(
        egui::pos2(center_x, value_y),
        egui::Align2::CENTER_TOP,
        value,
        egui::FontId::new(14.0, egui::FontFamily::Proportional),
        accent_text,
    );
    ui.painter().text(
        egui::pos2(center_x, subtitle_y),
        egui::Align2::CENTER_TOP,
        subtitle,
        egui::FontId::new(10.0, egui::FontFamily::Proportional),
        ui.visuals().weak_text_color(),
    );
}

fn render_responsive_two_pane(
    ui: &mut egui::Ui,
    id: &'static str,
    min_column_width: f32,
    left: impl FnOnce(&mut egui::Ui),
    right: impl FnOnce(&mut egui::Ui),
) {
    let spacing = ui.spacing().item_spacing.x.max(12.0);
    if ui.available_width() >= (min_column_width * 2.0) + spacing {
        ui.columns(2, |columns| {
            columns[0].set_min_width(min_column_width);
            left(&mut columns[0]);
            columns[1].set_min_width(min_column_width);
            right(&mut columns[1]);
        });
    } else {
        left(ui);
        ui.add_space(12.0);
        ui.push_id(id, |ui| {
            right(ui);
        });
    }
}

fn sidebar_card(ui: &mut egui::Ui, title: Option<&str>, add_contents: impl FnOnce(&mut egui::Ui)) {
    let palette = controller_palette_for_ui(ui);
    let width = ui.available_width();
    ui.allocate_ui_with_layout(
        egui::vec2(width, 0.0),
        egui::Layout::top_down(egui::Align::Min),
        |ui| {
            ui.set_width(width);
            ui.set_min_width(width);
            egui::Frame::default()
                .fill(palette.card_fill)
                .stroke(egui::Stroke::new(1.0, palette.border))
                .corner_radius(egui::CornerRadius::same(14))
                .inner_margin(egui::Margin::same(8))
                .show(ui, |ui| {
                    let inner_width = ui.available_width();
                    ui.set_width(inner_width);
                    ui.set_min_width(inner_width);
                    ui.with_layout(egui::Layout::top_down(egui::Align::Min), |ui| {
                        if let Some(title) = title {
                            ui.label(egui::RichText::new(title).strong().size(14.0));
                            ui.add_space(4.0);
                        }
                        add_contents(ui);
                    });
                });
        },
    );
}

fn card(ui: &mut egui::Ui, title: Option<&str>, add_contents: impl FnOnce(&mut egui::Ui)) {
    let palette = controller_palette_for_ui(ui);
    let width = ui.available_width();
    ui.allocate_ui_with_layout(
        egui::vec2(width, 0.0),
        egui::Layout::top_down(egui::Align::Min),
        |ui| {
            ui.set_width(width);
            ui.set_min_width(width);
            egui::Frame::default()
                .fill(palette.card_fill)
                .stroke(egui::Stroke::new(1.0, palette.border))
                .corner_radius(egui::CornerRadius::same(14))
                .inner_margin(egui::Margin::same(12))
                .show(ui, |ui| {
                    let inner_width = ui.available_width();
                    ui.set_width(inner_width);
                    ui.set_min_width(inner_width);
                    ui.with_layout(egui::Layout::top_down(egui::Align::Min), |ui| {
                        if let Some(title) = title {
                            ui.label(egui::RichText::new(title).strong().size(16.0));
                            ui.add_space(6.0);
                        }
                        add_contents(ui);
                    });
                });
        },
    );
}

fn outlined_card(ui: &mut egui::Ui, add_contents: impl FnOnce(&mut egui::Ui)) {
    let palette = controller_palette_for_ui(ui);
    let width = ui.available_width();
    ui.allocate_ui_with_layout(
        egui::vec2(width, 0.0),
        egui::Layout::top_down(egui::Align::Min),
        |ui| {
            ui.set_width(width);
            ui.set_min_width(width);
            egui::Frame::default()
                .fill(palette.outlined_card_fill)
                .stroke(egui::Stroke::new(1.0, palette.border_soft))
                .corner_radius(egui::CornerRadius::same(12))
                .inner_margin(egui::Margin::same(10))
                .show(ui, |ui| {
                    let inner_width = ui.available_width();
                    ui.set_width(inner_width);
                    ui.set_min_width(inner_width);
                    ui.with_layout(egui::Layout::top_down(egui::Align::Min), |ui| {
                        add_contents(ui);
                    });
                });
        },
    );
}

fn warning_card(ui: &mut egui::Ui, title: &str, body: &str) {
    let palette = controller_palette_for_ui(ui);
    egui::Frame::default()
        .fill(palette.warning_fill)
        .stroke(egui::Stroke::new(1.0, palette.warning_border))
        .corner_radius(egui::CornerRadius::same(12))
        .inner_margin(egui::Margin::same(10))
        .show(ui, |ui| {
            let inner_width = ui.available_width();
            ui.set_width(inner_width);
            ui.set_min_width(inner_width);
            ui.with_layout(egui::Layout::top_down(egui::Align::Min), |ui| {
                ui.label(
                    egui::RichText::new(title)
                        .strong()
                        .color(palette.warning_text),
                );
                ui.label(body);
            });
        });
}

fn render_readme_preview(
    app: &mut ClusterControllerApp,
    ui: &mut egui::Ui,
    id: &str,
    text: &str,
    max_height: f32,
) {
    ui.style_mut().url_in_tooltip = true;
    if max_height <= 0.0 {
        CommonMarkViewer::new().show(ui, &mut app.readme_markdown_cache, text);
        return;
    }
    egui::ScrollArea::vertical()
        .id_salt(id)
        .max_height(max_height)
        .show(ui, |ui| {
            CommonMarkViewer::new().show(ui, &mut app.readme_markdown_cache, text);
        });
}

fn render_about_document(_app: &mut ClusterControllerApp, ui: &mut egui::Ui, text: &str) {
    ui.add(
        egui::Label::new(egui::RichText::new(text).monospace())
            .selectable(true)
            .wrap(),
    );
}

fn adaptive_field_width(ui: &egui::Ui, fraction: f32, min_width: f32, max_width: f32) -> f32 {
    let available = ui.available_width().max(1.0);
    let desired = (available * fraction).clamp(min_width, max_width);
    desired.min(available)
}

fn adaptive_combo_width(ui: &egui::Ui, fraction: f32, min_width: f32, max_width: f32) -> f32 {
    adaptive_field_width(ui, fraction, min_width, max_width)
}

fn compact_control_layout(ui: &egui::Ui) -> bool {
    ui.available_width() < 700.0
}

fn discovery_seconds_remaining(expires_unix_ms: u64) -> u64 {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    expires_unix_ms.saturating_sub(now).div_ceil(1000)
}

fn download_progress_summary(progress: &crate::model_store::DownloadProgress) -> String {
    let mut parts = vec![
        format!(
            "{} / {} files",
            progress.completed_files,
            progress.total_files.max(progress.completed_files)
        ),
        if progress.total_bytes > 0 {
            format!(
                "{} / {}",
                format_bytes_compact(progress.downloaded_bytes),
                format_bytes_compact(progress.total_bytes)
            )
        } else {
            format_bytes_compact(progress.downloaded_bytes)
        },
    ];
    if progress.bytes_per_second > 0 {
        parts.push(format!(
            "{}/s",
            format_bytes_compact(progress.bytes_per_second)
        ));
    }
    if let Some(eta) = progress.eta_seconds {
        parts.push(format!("ETA {}", format_eta_compact(eta)));
    }
    parts.join(" | ")
}

fn format_eta_compact(seconds: u64) -> String {
    if seconds < 60 {
        return format!("{seconds}s");
    }
    if seconds < 3600 {
        return format!("{}m {}s", seconds / 60, seconds % 60);
    }
    let hours = seconds / 3600;
    let minutes = (seconds % 3600) / 60;
    if minutes == 0 {
        format!("{hours}h")
    } else {
        format!("{hours}h {minutes}m")
    }
}

fn summary_pill(
    ui: &mut egui::Ui,
    text: impl AsRef<str>,
    fill: egui::Color32,
    color: egui::Color32,
) {
    let (fill, color) = themed_badge_colors(ui, fill, color);
    let stroke = if ui.ctx().theme() == egui::Theme::Dark {
        egui::Stroke::new(1.0, blend_color(fill, color, 0.35))
    } else {
        egui::Stroke::NONE
    };
    egui::Frame::default()
        .fill(fill)
        .stroke(stroke)
        .corner_radius(egui::CornerRadius::same(255))
        .inner_margin(egui::Margin::symmetric(10, 4))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(text.as_ref()).strong().color(color));
            });
        });
}

fn state_badge(
    ui: &mut egui::Ui,
    text: impl AsRef<str>,
    fill: egui::Color32,
    color: egui::Color32,
) {
    summary_pill(ui, text, fill, color);
}

fn muted_label(ui: &mut egui::Ui, text: &str) {
    ui.label(egui::RichText::new(text).color(ui.visuals().weak_text_color()));
}

fn wrapped_muted_text(ui: &mut egui::Ui, text: &str) {
    ui.add(
        egui::Label::new(
            egui::RichText::new(text)
                .monospace()
                .color(ui.visuals().weak_text_color()),
        )
        .wrap(),
    );
}

fn wrapped_monospace(ui: &mut egui::Ui, text: &str) {
    ui.add(egui::Label::new(egui::RichText::new(text).monospace()).wrap());
}

fn accent_button(ui: &mut egui::Ui, label: &str) -> egui::Response {
    let palette = controller_palette_for_ui(ui);
    ui.add(
        egui::Button::new(
            egui::RichText::new(label)
                .strong()
                .color(egui::Color32::WHITE),
        )
        .fill(palette.accent_fill)
        .stroke(egui::Stroke::new(1.0, palette.accent_stroke))
        .corner_radius(egui::CornerRadius::same(10))
        .min_size(egui::vec2(0.0, 32.0)),
    )
}

fn secondary_button(ui: &mut egui::Ui, label: &str) -> egui::Response {
    let palette = controller_palette_for_ui(ui);
    ui.add(
        egui::Button::new(label)
            .fill(palette.secondary_fill)
            .stroke(egui::Stroke::new(1.0, palette.secondary_stroke))
            .corner_radius(egui::CornerRadius::same(10))
            .min_size(egui::vec2(0.0, 32.0)),
    )
}

fn secondary_button_enabled(ui: &mut egui::Ui, label: &str, enabled: bool) -> egui::Response {
    let palette = controller_palette_for_ui(ui);
    ui.add_enabled(
        enabled,
        egui::Button::new(label)
            .fill(palette.secondary_fill)
            .stroke(egui::Stroke::new(1.0, palette.secondary_stroke))
            .corner_radius(egui::CornerRadius::same(10))
            .min_size(egui::vec2(0.0, 32.0)),
    )
}

fn warning_button(ui: &mut egui::Ui, label: &str) -> egui::Response {
    ui.add(
        egui::Button::new(
            egui::RichText::new(label)
                .strong()
                .color(egui::Color32::WHITE),
        )
        .fill(egui::Color32::from_rgb(220, 38, 38))
        .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(153, 27, 27)))
        .corner_radius(egui::CornerRadius::same(10))
        .min_size(egui::vec2(0.0, 32.0)),
    )
}

fn nav_button(ui: &mut egui::Ui, label: &str, selected: bool) -> egui::Response {
    let palette = controller_palette_for_ui(ui);
    let fill = if selected {
        palette.nav_selected_fill
    } else {
        palette.card_fill
    };
    let stroke = if selected {
        egui::Stroke::new(1.0, palette.nav_selected_stroke)
    } else {
        egui::Stroke::new(1.0, palette.border)
    };
    let text = if selected {
        egui::RichText::new(label)
            .strong()
            .color(egui::Color32::WHITE)
    } else {
        egui::RichText::new(label).strong()
    };
    ui.add(
        egui::Button::new(text)
            .fill(fill)
            .stroke(stroke)
            .corner_radius(egui::CornerRadius::same(12))
            .min_size(egui::vec2(ui.available_width(), 36.0)),
    )
}

fn header_tab_button(ui: &mut egui::Ui, label: &str, selected: bool) -> egui::Response {
    let palette = controller_palette_for_ui(ui);
    let fill = if selected {
        palette.nav_selected_fill
    } else {
        palette.card_fill
    };
    let stroke = if selected {
        egui::Stroke::new(1.0, palette.nav_selected_stroke)
    } else {
        egui::Stroke::new(1.0, palette.border)
    };
    let text = if selected {
        egui::RichText::new(label)
            .strong()
            .color(egui::Color32::WHITE)
    } else {
        egui::RichText::new(label).strong()
    };
    ui.add(
        egui::Button::new(text)
            .fill(fill)
            .stroke(stroke)
            .corner_radius(egui::CornerRadius::same(12))
            .min_size(egui::vec2(98.0, 34.0)),
    )
}

fn header_action_button(ui: &mut egui::Ui, label: &str, busy: bool) -> egui::Response {
    let palette = controller_palette_for_ui(ui);
    let (fill, stroke, text) = if busy {
        (
            palette.accent_fill,
            egui::Stroke::new(1.0, palette.accent_stroke),
            egui::RichText::new(label)
                .strong()
                .color(egui::Color32::WHITE),
        )
    } else {
        (
            palette.secondary_fill,
            egui::Stroke::new(1.0, palette.secondary_stroke),
            egui::RichText::new(label).strong(),
        )
    };
    ui.add(
        egui::Button::new(text)
            .fill(fill)
            .stroke(stroke)
            .corner_radius(egui::CornerRadius::same(12))
            .min_size(egui::vec2(98.0, 34.0)),
    )
}

fn subtab_button(ui: &mut egui::Ui, label: &str, selected: bool) -> egui::Response {
    let palette = controller_palette_for_ui(ui);
    let fill = if selected {
        palette.subtab_selected_fill
    } else {
        palette.secondary_fill
    };
    let stroke = if selected {
        egui::Stroke::new(1.0, palette.subtab_selected_stroke)
    } else {
        egui::Stroke::new(1.0, palette.secondary_stroke)
    };
    let text = if selected {
        egui::RichText::new(label)
            .strong()
            .color(egui::Color32::WHITE)
    } else {
        egui::RichText::new(label).strong()
    };
    ui.add(
        egui::Button::new(text)
            .fill(fill)
            .stroke(stroke)
            .corner_radius(egui::CornerRadius::same(10))
            .min_size(egui::vec2(144.0, 32.0)),
    )
}

fn cluster_loaded_instances(app: &ClusterControllerApp) -> usize {
    app.telemetry
        .iter()
        .map(|snapshot| {
            snapshot
                .instances
                .iter()
                .filter(|instance| instance.state != 0)
                .count()
        })
        .sum()
}

fn cluster_serving_instances(app: &ClusterControllerApp) -> usize {
    app.telemetry
        .iter()
        .map(|snapshot| {
            snapshot
                .instances
                .iter()
                .filter(|instance| instance.active_request_count > 0 || instance.state == 3)
                .count()
        })
        .sum()
}

fn cluster_queued_requests(app: &ClusterControllerApp) -> usize {
    app.telemetry
        .iter()
        .map(|snapshot| {
            snapshot
                .instances
                .iter()
                .map(|instance| instance.queued_request_count.max(0) as usize)
                .sum::<usize>()
        })
        .sum()
}

fn total_free_gpu_memory(app: &ClusterControllerApp) -> u64 {
    app.telemetry
        .iter()
        .filter_map(|snapshot| {
            app.nodes
                .iter()
                .find(|node| node.control_addr == snapshot.control_addr)
                .map(|node| (node, snapshot))
        })
        .flat_map(|(node, snapshot)| filtered_devices_for_node(app, node, Some(snapshot)))
        .filter(|device| !is_cpu_device(device) && !is_rpc_device(device))
        .map(|device| device.memory_free)
        .sum()
}

#[derive(Clone)]
struct LinkSpeedWidgetRow {
    pair_key: String,
    pair_label: String,
    speed_label: String,
    subtitle: String,
    hover_text: String,
    unix_ms: u64,
    goodput_mbps: f64,
    successful: bool,
    manual: bool,
    pending: bool,
}

fn render_link_speed_overview_widgets(app: &mut ClusterControllerApp, ui: &mut egui::Ui) {
    let rows = link_speed_widget_rows(app);
    let has_rows = !rows.is_empty();
    let rows = if has_rows {
        rows
    } else {
        vec![LinkSpeedWidgetRow {
            pair_key: "__placeholder__".to_string(),
            pair_label: "Link speed".to_string(),
            speed_label: if app.link_benchmark_in_progress {
                "Running".to_string()
            } else {
                "No probe".to_string()
            },
            subtitle: if app.link_benchmark_in_progress {
                "Benchmark in progress".to_string()
            } else {
                "Waiting for the first paired-node measurement".to_string()
            },
            hover_text: if app.link_benchmark_in_progress {
                "A sustained benchmark is running in the background.".to_string()
            } else {
                "Link speeds are measured once when a node connection is first established or re-established, and again only when you rerun the benchmark manually.".to_string()
            },
            unix_ms: 0,
            goodput_mbps: 0.0,
            successful: false,
            manual: false,
            pending: !app.link_benchmark_in_progress,
        }]
    };
    for (index, entry) in rows.iter().enumerate() {
        let accent = if entry.pending {
            egui::Color32::from_rgb(217, 119, 6)
        } else if entry.successful {
            egui::Color32::from_rgb(14, 116, 144)
        } else {
            egui::Color32::from_rgb(185, 28, 28)
        };
        let base_response = summary_card_sized_response(
            ui,
            egui::vec2(ui.available_width(), 56.0),
            &entry.pair_label,
            &entry.speed_label,
            &entry.subtitle,
            accent,
        );
        let response = ui
            .interact(
                base_response.rect,
                ui.id()
                    .with(("link-speed-widget", index, entry.pair_key.as_str())),
                egui::Sense::click(),
            )
            .on_hover_text(&entry.hover_text);
        response.context_menu(|ui| {
            if ui.button("Rerun sustained benchmark").clicked() {
                app.run_cluster_link_benchmarks(true);
                ui.close();
            }
            if !has_rows {
                muted_label(ui, "No saved benchmark yet. Run one to populate this card.");
            }
        });
        if index + 1 < rows.len() {
            ui.add_space(3.0);
        }
    }
}

fn link_speed_widget_rows(app: &ClusterControllerApp) -> Vec<LinkSpeedWidgetRow> {
    let mut name_by_addr = BTreeMap::new();
    for snapshot in &app.telemetry {
        name_by_addr.insert(
            snapshot.control_addr.clone(),
            snapshot.node.display_name.clone(),
        );
        if let Some(advertised) = &snapshot.advertised_control_addr {
            name_by_addr.insert(advertised.clone(), snapshot.node.display_name.clone());
        }
    }
    for node in &app.nodes {
        name_by_addr.insert(node.control_addr.clone(), node.node.display_name.clone());
        if let Some(advertised) = &node.advertised_control_addr {
            name_by_addr.insert(advertised.clone(), node.node.display_name.clone());
        }
    }

    let mut best_by_pair = BTreeMap::new();
    let mut known_pairs = BTreeMap::new();
    for snapshot in &app.telemetry {
        let source_addr = snapshot.control_addr.clone();
        let source_label = name_by_addr
            .get(&source_addr)
            .cloned()
            .unwrap_or_else(|| snapshot.node.display_name.clone());
        for link in &snapshot.link_metrics {
            if link.peer_control_addr.trim().is_empty() || link.peer_control_addr == source_addr {
                continue;
            }
            let mut pair_addrs = [source_addr.clone(), link.peer_control_addr.clone()];
            pair_addrs.sort();
            let pair_key = format!("{}|{}", pair_addrs[0], pair_addrs[1]);
            let peer_label = name_by_addr
                .get(&link.peer_control_addr)
                .cloned()
                .unwrap_or_else(|| link.peer_control_addr.clone());
            known_pairs.entry(pair_key.clone()).or_insert_with(|| {
                (
                    source_label.clone(),
                    peer_label.clone(),
                    link.transport.clone(),
                )
            });
            let candidate =
                link_speed_widget_row(pair_key.clone(), source_label.clone(), peer_label, link);
            match best_by_pair.get(&pair_key) {
                Some(existing) if !prefer_link_speed_row(&candidate, existing) => {}
                _ => {
                    best_by_pair.insert(pair_key, candidate);
                }
            }
        }
    }

    for source in &app.nodes {
        for peer in &app.nodes {
            if source.control_addr >= peer.control_addr {
                continue;
            }
            let pair_key = format!("{}|{}", source.control_addr, peer.control_addr);
            known_pairs.entry(pair_key).or_insert_with(|| {
                (
                    source.node.display_name.clone(),
                    peer.node.display_name.clone(),
                    link_transport_label_for_addr(&peer.control_addr),
                )
            });
        }
    }

    for (pair_key, (source_label, peer_label, transport)) in known_pairs {
        best_by_pair
            .entry(pair_key.clone())
            .or_insert(LinkSpeedWidgetRow {
                pair_key,
                pair_label: format!("{source_label} ↔ {peer_label}"),
                speed_label: if app.link_benchmark_in_progress {
                    "Running".to_string()
                } else {
                    "No probe".to_string()
                },
                subtitle: if app.link_benchmark_in_progress {
                    transport.clone()
                } else {
                    format!("{transport} | waiting")
                },
                hover_text: if app.link_benchmark_in_progress {
                    format!("{transport} | waiting for the current benchmark run to finish")
                } else {
                    format!("{transport} | no benchmark recorded for this pair yet")
                },
                unix_ms: 0,
                goodput_mbps: 0.0,
                successful: false,
                manual: false,
                pending: true,
            });
    }

    let mut rows = best_by_pair.into_values().collect::<Vec<_>>();
    rows.sort_by(|lhs, rhs| {
        lhs.pending.cmp(&rhs.pending).then_with(|| {
            rhs.successful
                .cmp(&lhs.successful)
                .then_with(|| rhs.manual.cmp(&lhs.manual))
                .then_with(|| rhs.unix_ms.cmp(&lhs.unix_ms))
                .then_with(|| rhs.goodput_mbps.total_cmp(&lhs.goodput_mbps))
                .then_with(|| lhs.pair_key.cmp(&rhs.pair_key))
        })
    });
    rows
}

fn link_speed_widget_row(
    pair_key: String,
    source_label: String,
    peer_label: String,
    link: &LinkMetrics,
) -> LinkSpeedWidgetRow {
    let manual = link.probe_kind.eq_ignore_ascii_case("manual");
    let successful = link.error.is_none();
    let probe_time = format_probe_clock_label(link.unix_ms);
    let speed_label = match &link.error {
        Some(_) => "benchmark failed".to_string(),
        None if link.goodput_mbps >= 1_000.0 => {
            format!(
                "{:.2} Gbps | {}",
                link.goodput_mbps / 1_000.0,
                format_latency_label(link.latency_ms)
            )
        }
        None => format!(
            "{:.0} Mbps | {}",
            link.goodput_mbps,
            format_latency_label(link.latency_ms)
        ),
    };
    let subtitle = match &link.error {
        Some(_) => format!("{} | failed", link.transport),
        None => format!(
            "{} | {} | {}",
            link.transport,
            if manual { "manual" } else { "auto" },
            probe_time
        ),
    };
    let hover_text = match &link.error {
        Some(error) => format!(
            "{} | {} | {} | {}",
            link.transport,
            if manual { "manual" } else { "startup" },
            probe_time,
            error
        ),
        None => format!(
            "{} | {} | {} | {} | latency {} | duration {:.0} ms",
            link.transport,
            if manual { "manual" } else { "startup" },
            format_bytes_compact(link.payload_bytes),
            probe_time,
            format_latency_label(link.latency_ms),
            link.duration_ms
        ),
    };
    LinkSpeedWidgetRow {
        pair_key,
        pair_label: format!("{source_label} ↔ {peer_label}"),
        speed_label,
        subtitle,
        hover_text,
        unix_ms: link.unix_ms,
        goodput_mbps: link.goodput_mbps,
        successful,
        manual,
        pending: false,
    }
}

fn format_latency_label(latency_ms: f64) -> String {
    if !latency_ms.is_finite() || latency_ms <= 0.0 {
        return "-- ms".to_string();
    }
    if latency_ms < 10.0 {
        format!("{latency_ms:.1} ms")
    } else {
        format!("{latency_ms:.0} ms")
    }
}

fn format_probe_clock_label(unix_ms: u64) -> String {
    let Ok(timestamp) = OffsetDateTime::from_unix_timestamp_nanos((unix_ms as i128) * 1_000_000)
    else {
        return "--:--".to_string();
    };
    let offset = UtcOffset::current_local_offset().unwrap_or(UtcOffset::UTC);
    let local = timestamp.to_offset(offset);
    format!("{:02}:{:02}", local.hour(), local.minute())
}

fn prefer_link_speed_row(candidate: &LinkSpeedWidgetRow, existing: &LinkSpeedWidgetRow) -> bool {
    candidate.pending.cmp(&existing.pending).is_lt()
        || (candidate.pending == existing.pending
            && (candidate.unix_ms > existing.unix_ms
                || (candidate.unix_ms == existing.unix_ms
                    && (candidate.successful.cmp(&existing.successful).is_gt()
                        || (candidate.successful == existing.successful
                            && candidate.goodput_mbps > existing.goodput_mbps)))))
}

fn link_transport_label_for_addr(control_addr: &str) -> String {
    let host = control_addr
        .trim()
        .trim_start_matches('[')
        .split(']')
        .next()
        .unwrap_or(control_addr)
        .split(':')
        .next()
        .unwrap_or(control_addr)
        .trim();
    if host.starts_with("169.254.") || host.starts_with("fe80:") {
        "thunderbolt/link-local".to_string()
    } else if host.starts_with("10.")
        || host.starts_with("192.168.")
        || host.starts_with("172.16.")
        || host.starts_with("172.17.")
        || host.starts_with("172.18.")
        || host.starts_with("172.19.")
        || host.starts_with("172.2")
        || host.starts_with("172.30.")
        || host.starts_with("172.31.")
    {
        "lan".to_string()
    } else {
        "network".to_string()
    }
}

fn node_loaded_instances(node: &NodeSnapshot, telemetry: Option<&TelemetrySnapshot>) -> usize {
    telemetry
        .map(|snapshot| {
            snapshot
                .instances
                .iter()
                .filter(|instance| instance.state != 0)
                .count()
        })
        .unwrap_or_else(|| {
            node.instances
                .iter()
                .filter(|instance| instance.state != 0)
                .count()
        })
}

fn instance_model_type_options() -> [(&'static str, &'static str); 6] {
    [
        ("", "All models"),
        ("chat", "Chat"),
        ("vision", "Vision"),
        ("embeddings", "Embeddings"),
        ("rerank", "Rerank"),
        ("transcription", "Audio / Transcription"),
    ]
}

fn instance_model_type_label(value: &str) -> &'static str {
    instance_model_type_options()
        .into_iter()
        .find(|(candidate, _)| *candidate == value)
        .map(|(_, label)| label)
        .unwrap_or("All models")
}

fn model_matches_instance_type(model: &ManagedModelEntry, value: &str) -> bool {
    match value {
        "" => true,
        "chat" => model.task == ManagedModelTask::Responses && !model.supports_vision(),
        "vision" => model.supports_vision(),
        "embeddings" => model.task == ManagedModelTask::Embeddings,
        "rerank" => model.task == ManagedModelTask::Rerank,
        "transcription" => model.task == ManagedModelTask::Transcription,
        _ => true,
    }
}

fn instance_creation_models(app: &ClusterControllerApp) -> Vec<ManagedModelEntry> {
    let mut models = app
        .managed_models
        .iter()
        .filter(|model| model_matches_instance_type(model, &app.model_family_filter))
        .cloned()
        .collect::<Vec<_>>();
    models.sort_by(|lhs, rhs| lhs.display_name.cmp(&rhs.display_name));
    models
}

fn filtered_models(app: &ClusterControllerApp) -> Vec<ManagedModelEntry> {
    let search = app.model_search.trim().to_ascii_lowercase();
    app.managed_models
        .iter()
        .filter(|model| {
            (app.model_family_filter.is_empty() || model.family == app.model_family_filter)
                && (search.is_empty()
                    || model.id.to_ascii_lowercase().contains(&search)
                    || model.display_name.to_ascii_lowercase().contains(&search)
                    || model.family.to_ascii_lowercase().contains(&search)
                    || task_label(model.task).contains(&search)
                    || model
                        .diarization_model_path
                        .as_deref()
                        .unwrap_or_default()
                        .to_ascii_lowercase()
                        .contains(&search))
        })
        .cloned()
        .collect()
}

fn model_families(models: &[ManagedModelEntry]) -> Vec<String> {
    let mut families = models
        .iter()
        .map(|model| model.family.clone())
        .collect::<Vec<_>>();
    families.sort();
    families.dedup();
    families
}

fn task_label(task: ManagedModelTask) -> &'static str {
    match task {
        ManagedModelTask::Responses => "responses",
        ManagedModelTask::Embeddings => "embeddings",
        ManagedModelTask::Rerank => "rerank",
        ManagedModelTask::Transcription => "transcription",
    }
}

fn model_available_node_labels(
    app: &ClusterControllerApp,
    model: &ManagedModelEntry,
) -> Vec<String> {
    let mut labels = model
        .allowed_control_addrs
        .clone()
        .unwrap_or_else(|| app.default_allowed_node_addrs().into_iter().collect())
        .into_iter()
        .filter_map(|addr| {
            app.nodes
                .iter()
                .find(|node| {
                    node.control_addr == addr
                        || node
                            .advertised_control_addr
                            .as_deref()
                            .is_some_and(|value| value == addr)
                })
                .map(|node| node.node.display_name.clone())
                .or(Some(addr))
        })
        .collect::<Vec<_>>();
    labels.sort();
    labels.dedup();
    labels
}

fn render_instance_slot_summary(ui: &mut egui::Ui, instance: &crate::cluster_api::InstanceInfo) {
    let slots = instance.n_parallel.max(1);
    let active = instance.active_request_count.max(0);
    let queued = instance.queued_request_count.max(0);
    ui.add(
        egui::ProgressBar::new((active as f32 / slots as f32).clamp(0.0, 1.0))
            .desired_width(ui.available_width())
            .text(format!(
                "Slots in use {} / {} | queued {}",
                active, slots, queued
            )),
    );
    let available = (slots - active).max(0);
    muted_label(
        ui,
        &format!("Ready slots now: {} | queue depth: {}", available, queued),
    );
}

fn setup_check_item(ui: &mut egui::Ui, ready: bool, title: &str, body: &str) {
    outlined_card(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            state_badge(
                ui,
                if ready { "ready" } else { "action needed" },
                if ready {
                    egui::Color32::from_rgb(220, 252, 231)
                } else {
                    egui::Color32::from_rgb(254, 226, 226)
                },
                if ready {
                    egui::Color32::from_rgb(22, 101, 52)
                } else {
                    egui::Color32::from_rgb(153, 27, 27)
                },
            );
            ui.label(egui::RichText::new(title).strong());
        });
        muted_label(ui, body);
    });
}

fn cluster_instances(
    app: &ClusterControllerApp,
) -> Vec<(String, String, crate::cluster_api::InstanceInfo)> {
    let mut out = Vec::new();
    let live_by_addr = app
        .telemetry
        .iter()
        .map(|snapshot| (snapshot.control_addr.clone(), snapshot.instances.clone()))
        .collect::<BTreeMap<_, _>>();
    for node in &app.nodes {
        let instances = live_by_addr
            .get(&node.control_addr)
            .cloned()
            .unwrap_or_else(|| node.instances.clone());
        for instance in instances {
            out.push((
                node.control_addr.clone(),
                node.node.display_name.clone(),
                instance,
            ));
        }
    }
    out.sort_by(|lhs, rhs| {
        lhs.1
            .cmp(&rhs.1)
            .then(lhs.2.name.cmp(&rhs.2.name))
            .then(lhs.2.instance_id.cmp(&rhs.2.instance_id))
    });
    out
}

fn state_fill(state: i32) -> egui::Color32 {
    match state {
        0 => egui::Color32::from_rgb(241, 245, 249),
        1 => egui::Color32::from_rgb(254, 249, 195),
        2 => egui::Color32::from_rgb(220, 252, 231),
        3 => egui::Color32::from_rgb(224, 242, 254),
        4 => egui::Color32::from_rgb(243, 232, 255),
        5 => egui::Color32::from_rgb(254, 226, 226),
        _ => egui::Color32::from_rgb(241, 245, 249),
    }
}

fn state_color(state: i32) -> egui::Color32 {
    match state {
        0 => egui::Color32::from_rgb(71, 85, 105),
        1 => egui::Color32::from_rgb(133, 77, 14),
        2 => egui::Color32::from_rgb(22, 101, 52),
        3 => egui::Color32::from_rgb(14, 116, 144),
        4 => egui::Color32::from_rgb(107, 33, 168),
        5 => egui::Color32::from_rgb(153, 27, 27),
        _ => egui::Color32::from_rgb(71, 85, 105),
    }
}

fn retention_label(mode: RetentionMode) -> &'static str {
    match mode {
        RetentionMode::KeepLoaded => "keep loaded",
        RetentionMode::LoadOnDemand => "load on demand",
    }
}

fn render_load_on_demand_grace_editor(ui: &mut egui::Ui, app: &mut ClusterControllerApp) {
    if app.create_params.retention_mode != RetentionMode::LoadOnDemand {
        return;
    }
    ui.label("Grace");
    ui.add(
        egui::DragValue::new(&mut app.create_params.load_on_demand_grace_seconds)
            .range(0..=604800)
            .speed(1),
    );
    ui.label("sec");
}

fn load_on_demand_grace_summary_suffix(retention_mode: RetentionMode, seconds: i32) -> String {
    if retention_mode != RetentionMode::LoadOnDemand {
        return String::new();
    }
    if seconds <= 0 {
        " | immediate unload".to_string()
    } else {
        format!(" | grace {} sec", seconds)
    }
}

fn yes_no(value: bool) -> &'static str {
    if value {
        "yes"
    } else {
        "no"
    }
}

fn memory_ratio(used: u64, total: u64) -> f32 {
    if total == 0 {
        0.0
    } else {
        (used as f64 / total as f64).clamp(0.0, 1.0) as f32
    }
}

fn parse_bind_addr_parts(value: &str) -> Option<(String, String)> {
    let (host, port) = value.trim().rsplit_once(':')?;
    if host.is_empty() || port.is_empty() {
        return None;
    }
    Some((host.to_string(), port.to_string()))
}

fn server_bind_host_option_label(host: &str, options: &[PublicApiBindHostOption]) -> String {
    options
        .iter()
        .find(|option| option.host == host.trim())
        .map(|option| option.label.clone())
        .unwrap_or_else(|| host.trim().to_string())
}

fn server_scope_label(bind_addr: &str) -> &'static str {
    let Some((host, _)) = parse_bind_addr_parts(bind_addr) else {
        return "unknown";
    };
    match host.trim() {
        "127.0.0.1" | "localhost" | "::1" | "[::1]" => "this machine only",
        _ => "local network only",
    }
}

fn configure_preferred_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    let mut changed = false;

    #[cfg(target_os = "windows")]
    {
        changed |= load_font_family(
            &mut fonts,
            "segoe_ui",
            &[Path::new(r"C:\Windows\Fonts\segoeui.ttf")],
            egui::FontFamily::Proportional,
        );
        changed |= load_font_family(
            &mut fonts,
            "consolas",
            &[Path::new(r"C:\Windows\Fonts\consola.ttf")],
            egui::FontFamily::Monospace,
        );
    }

    #[cfg(target_os = "macos")]
    {
        changed |= load_font_family(
            &mut fonts,
            "sf_pro",
            &[
                Path::new("/System/Library/Fonts/SFNS.ttf"),
                Path::new("/System/Library/Fonts/Helvetica.ttc"),
            ],
            egui::FontFamily::Proportional,
        );
        changed |= load_font_family(
            &mut fonts,
            "sf_mono",
            &[
                Path::new("/System/Library/Fonts/SFNSMono.ttf"),
                Path::new("/System/Library/Fonts/Menlo.ttc"),
                Path::new("/System/Library/Fonts/Monaco.ttf"),
            ],
            egui::FontFamily::Monospace,
        );
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        changed |= load_font_family(
            &mut fonts,
            "ubuntu",
            &[
                Path::new("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf"),
                Path::new("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            ],
            egui::FontFamily::Proportional,
        );
        changed |= load_font_family(
            &mut fonts,
            "ubuntu_mono",
            &[
                Path::new("/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf"),
                Path::new("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
            ],
            egui::FontFamily::Monospace,
        );
    }

    if changed {
        ctx.set_fonts(fonts);
    }
}

fn load_font_family(
    fonts: &mut egui::FontDefinitions,
    font_name: &str,
    candidates: &[&Path],
    family: egui::FontFamily,
) -> bool {
    let Some(path) = candidates.iter().find(|path| path.exists()) else {
        return false;
    };
    let Ok(data) = fs::read(path) else {
        return false;
    };
    fonts.font_data.insert(
        font_name.to_string(),
        std::sync::Arc::new(egui::FontData::from_owned(data)),
    );
    if let Some(fonts_for_family) = fonts.families.get_mut(&family) {
        fonts_for_family.insert(0, font_name.to_string());
    }
    true
}
